//! Long-running daemon: owns mic + camera (later milestones) and answers
//! control requests over the unix socket.

use crate::control::{self, Request, Response, State, Target};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};

pub struct Daemon {
    pub state: Mutex<State>,
    dictation: Mutex<Option<crate::dictation::Dictation>>,
    gaze: Mutex<Option<crate::gaze::Gaze>>,
}

pub fn run() -> std::io::Result<()> {
    let dir = control::state_dir();
    std::fs::create_dir_all(&dir)?;
    let sock = control::socket_path();

    // Refuse to double-start; replace a stale socket from a dead daemon.
    if control::daemon_alive() {
        eprintln!("herdr-handsfree daemon already running at {}", sock.display());
        return Ok(());
    }
    let _ = std::fs::remove_file(&sock);

    let listener = UnixListener::bind(&sock)?;
    eprintln!("herdr-handsfree daemon listening on {}", sock.display());

    let daemon = Arc::new(Daemon {
        state: Mutex::new(State::default()),
        dictation: Mutex::new(None),
        gaze: Mutex::new(None),
    });

    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let worker = daemon.clone();
        std::thread::spawn(move || {
            let _ = handle(&worker, stream);
        });
        if daemon.stopped() {
            break;
        }
    }
    let _ = std::fs::remove_file(&sock);
    Ok(())
}

impl Daemon {
    fn stopped(&self) -> bool {
        STOP.load(std::sync::atomic::Ordering::SeqCst)
    }
}

static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn handle(daemon: &Daemon, stream: UnixStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let resp = match serde_json::from_str::<Request>(&line) {
        Ok(req) => dispatch(daemon, req),
        Err(e) => Response {
            ok: false,
            state: daemon.state.lock().unwrap().clone(),
            error: Some(format!("bad request: {e}")),
            message: None,
        },
    };
    let mut out = serde_json::to_string(&resp).expect("serialize response");
    out.push('\n');
    let mut stream = stream;
    stream.write_all(out.as_bytes())
}

fn dispatch(daemon: &Daemon, req: Request) -> Response {
    let mut message = None;
    let mut error = None;
    match req {
        Request::Ping | Request::Status => {}
        Request::Toggle { target } => {
            let mut state = daemon.state.lock().unwrap();
            match target {
                Target::Dictation => state.dictation = !state.dictation,
                Target::Gaze => state.gaze = !state.gaze,
            }
            apply(daemon, &state);
        }
        Request::Stop => {
            STOP.store(true, std::sync::atomic::Ordering::SeqCst);
            // The accept loop notices on the next connection; nudge it.
            std::thread::spawn(|| {
                let _ = UnixStream::connect(control::socket_path());
            });
        }
        // Lock order everywhere: state before gaze (apply() holds state while
        // locking gaze, so any gaze→state nesting would deadlock).
        Request::CalStart => {
            let mut state = daemon.state.lock().unwrap();
            // The camera turns on for calibration; surface that in the flag so
            // an aborted calibration can be seen and toggled off.
            state.gaze = true;
            let mut gaze = daemon.gaze.lock().unwrap();
            let gaze = gaze.get_or_insert_with(|| crate::gaze::Gaze::start(false));
            gaze.shared
                .move_enabled
                .store(false, std::sync::atomic::Ordering::SeqCst);
            gaze.samples.lock().unwrap().clear();
        }
        Request::CalTarget { index } => {
            let [x, y] = crate::gaze::calibration_point(index);
            if let Err(e) = crate::gaze::mouse::move_cursor(x, y) {
                error = Some(e);
            }
        }
        Request::CalCapture { index } => {
            let gaze = daemon.gaze.lock().unwrap();
            match gaze.as_ref().and_then(|g| g.capture_features()) {
                Some(features) => {
                    let point = crate::gaze::calibration_point(index);
                    gaze.as_ref()
                        .unwrap()
                        .samples
                        .lock()
                        .unwrap()
                        .push((features, point));
                }
                None => {
                    error = Some(
                        "no face detected yet — face the camera and try again".to_string(),
                    );
                }
            }
        }
        Request::CalFinish => {
            let mut state = daemon.state.lock().unwrap();
            let gaze = daemon.gaze.lock().unwrap();
            match gaze.as_ref() {
                Some(g) => match g.finish_calibration() {
                    Ok(rms) => {
                        message = Some(format!("calibrated, fit rms {rms:.0} px"));
                        // Calibration implies the user wants gaze moves on.
                        g.shared
                            .move_enabled
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        state.gaze = true;
                    }
                    Err(e) => error = Some(e),
                },
                None => error = Some("calibration not started".to_string()),
            }
        }
    }
    let mut state = daemon.state.lock().unwrap().clone();
    state.gaze_calibrated = daemon
        .gaze
        .lock()
        .unwrap()
        .as_ref()
        .map(|g| g.is_calibrated())
        .unwrap_or_else(|| crate::control::state_dir().join("gaze-mapping.json").exists());
    Response {
        ok: error.is_none(),
        state,
        error,
        message,
    }
}

/// React to state changes: start/stop the pipelines to match the flags.
fn apply(daemon: &Daemon, state: &State) {
    eprintln!("state: dictation={} gaze={}", state.dictation, state.gaze);
    let mut dictation = daemon.dictation.lock().unwrap();
    match (state.dictation, dictation.is_some()) {
        (true, false) => *dictation = Some(crate::dictation::Dictation::start()),
        (false, true) => *dictation = None,
        _ => {}
    }
    let mut gaze = daemon.gaze.lock().unwrap();
    match (state.gaze, gaze.as_ref()) {
        (true, None) => *gaze = Some(crate::gaze::Gaze::start(true)),
        (true, Some(g)) => g
            .shared
            .move_enabled
            .store(true, std::sync::atomic::Ordering::SeqCst),
        // Drop the pipeline entirely so the camera turns off.
        (false, Some(_)) => *gaze = None,
        (false, None) => {}
    }
}
