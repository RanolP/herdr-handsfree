mod config;
mod control;
mod daemon;
mod deliver;
mod dictation;
mod gaze;
mod models;
mod silero;
mod stt;
mod vad;

use clap::{Parser, Subcommand};
use control::{Request, Target};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "herdr-handsfree", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the daemon in the foreground
    Daemon,
    /// Start the daemon in the background if it is not already running
    EnsureDaemon,
    /// Toggle dictation or gaze on/off
    Toggle { target: Target },
    /// Show daemon state
    Status {
        /// Redraw continuously (for the status pane)
        #[arg(long)]
        follow: bool,
    },
    /// Guided 9-point gaze calibration (milestone 3)
    Calibrate,
    /// Stop the daemon
    Stop,
    /// Offline self-check: run a WAV file through VAD + whisper, print the transcript
    #[command(hide = true)]
    TranscribeFile { path: std::path::PathBuf },
    /// Offline self-check: run an image through the gaze landmark pipeline
    #[command(hide = true)]
    GazeProbe { path: std::path::PathBuf },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Daemon => match daemon::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("daemon error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::EnsureDaemon => ensure_daemon(),
        Command::Toggle { target } => {
            if ensure_daemon() == ExitCode::FAILURE {
                return ExitCode::FAILURE;
            }
            roundtrip(&Request::Toggle { target })
        }
        Command::Status { follow } => status(follow),
        Command::Calibrate => calibrate(),
        Command::Stop => roundtrip(&Request::Stop),
        Command::TranscribeFile { path } => transcribe_file(&path),
        Command::GazeProbe { path } => gaze_probe(&path),
    }
}

fn gaze_probe(path: &std::path::Path) -> ExitCode {
    let img = match image::open(path) {
        Ok(i) => i.to_rgb8(),
        Err(e) => {
            eprintln!("cannot open {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut net = match gaze::landmarks::GazeNet::load() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match net.process(&img.into_raw(), w, h) {
        Some(features) => {
            println!("features: {features:?}");
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("no face detected");
            ExitCode::FAILURE
        }
    }
}

fn transcribe_file(path: &std::path::Path) -> ExitCode {
    let mut reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot open {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let spec = reader.spec();
    let mono: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    };
    let mono: Vec<f32> = mono
        .chunks(spec.channels as usize)
        .map(|f| f.iter().sum::<f32>() / f.len() as f32)
        .collect();

    let stt = match stt::Stt::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut resampler = vad::Resampler::new(spec.sample_rate as usize);
    let mut vad = vad::Vad::new();
    let mut utterances: Vec<Vec<f32>> = Vec::new();
    for chunk in mono.chunks(1024) {
        if let Some(u) = vad.push(&resampler.push(chunk)) {
            utterances.push(u);
        }
    }
    if let Some(u) = vad.finish() {
        utterances.push(u);
    }
    if utterances.is_empty() {
        eprintln!("no speech detected");
        return ExitCode::FAILURE;
    }
    for u in utterances {
        match stt.transcribe(&u) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

fn ensure_daemon() -> ExitCode {
    if control::daemon_alive() {
        return ExitCode::SUCCESS;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot locate own binary: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dir = control::state_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("cannot create state dir {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }
    let log = match std::fs::File::create(dir.join("daemon.log")) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot create daemon.log: {e}");
            return ExitCode::FAILURE;
        }
    };
    let spawned = std::process::Command::new(exe)
        .arg("daemon")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log)
        .spawn();
    if let Err(e) = spawned {
        eprintln!("cannot spawn daemon: {e}");
        return ExitCode::FAILURE;
    }
    // Wait briefly for the socket to come up.
    for _ in 0..50 {
        if control::daemon_alive() {
            println!("daemon started");
            return ExitCode::SUCCESS;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    eprintln!(
        "daemon did not come up within 5s; see {}",
        dir.join("daemon.log").display()
    );
    ExitCode::FAILURE
}

fn roundtrip(req: &Request) -> ExitCode {
    match control::request(req) {
        Ok(resp) if resp.ok => {
            println!(
                "dictation={} gaze={}",
                on_off(resp.state.dictation),
                on_off(resp.state.gaze)
            );
            ExitCode::SUCCESS
        }
        Ok(resp) => {
            eprintln!("daemon error: {}", resp.error.unwrap_or_default());
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("cannot reach daemon ({e}); try `herdr-handsfree ensure-daemon`");
            ExitCode::FAILURE
        }
    }
}

fn calibrate() -> ExitCode {
    if ensure_daemon() == ExitCode::FAILURE {
        return ExitCode::FAILURE;
    }
    if let Err(e) = control::request(&Request::CalStart) {
        eprintln!("cannot start calibration: {e}");
        return ExitCode::FAILURE;
    }
    println!("Gaze calibration: the mouse cursor will jump to 9 points.");
    println!("Look at the cursor each time, keep your head still, press Enter.\n");
    let stdin = std::io::stdin();
    for i in 0..9 {
        match control::request(&Request::CalTarget { index: i }) {
            Ok(r) if r.ok => {}
            Ok(r) => {
                eprintln!("cursor warp failed: {}", r.error.unwrap_or_default());
                return ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("daemon lost: {e}");
                return ExitCode::FAILURE;
            }
        }
        loop {
            print!("[{}/9] look at the cursor, then press Enter... ", i + 1);
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            if stdin.read_line(&mut line).is_err() {
                return ExitCode::FAILURE;
            }
            match control::request(&Request::CalCapture { index: i }) {
                Ok(r) if r.ok => break,
                Ok(r) => println!("  {} (retrying)", r.error.unwrap_or_default()),
                Err(e) => {
                    eprintln!("daemon lost: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    match control::request(&Request::CalFinish) {
        Ok(r) if r.ok => {
            println!("\n{}", r.message.unwrap_or_default());
            println!("Gaze mouse is now ON. Toggle it with the toggle-gaze action.");
            std::thread::sleep(std::time::Duration::from_secs(3));
            ExitCode::SUCCESS
        }
        Ok(r) => {
            eprintln!("calibration failed: {}", r.error.unwrap_or_default());
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("daemon lost: {e}");
            ExitCode::FAILURE
        }
    }
}

fn status(follow: bool) -> ExitCode {
    loop {
        let line = match control::request(&Request::Status) {
            Ok(resp) => format!(
                "handsfree  dictation: {}   gaze: {}{}",
                on_off(resp.state.dictation),
                on_off(resp.state.gaze),
                if resp.state.gaze_calibrated { "" } else { "  (not calibrated)" }
            ),
            Err(_) => "handsfree  daemon: not running".to_string(),
        };
        if follow {
            print!("\x1b[2J\x1b[H{line}\n\n(close this pane to dismiss)");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            std::thread::sleep(std::time::Duration::from_millis(500));
        } else {
            println!("{line}");
            return ExitCode::SUCCESS;
        }
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "ON" } else { "off" }
}
