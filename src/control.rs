//! Control protocol between thin action commands and the daemon.
//! Line-delimited JSON over a unix socket in the plugin state dir.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    Ping,
    Status,
    Toggle { target: Target },
    Stop,
    /// Start the gaze pipeline (without cursor moves) for calibration.
    CalStart,
    /// Warp the cursor to calibration point `index` (0..9).
    CalTarget { index: usize },
    /// Record the current gaze features against calibration point `index`.
    CalCapture { index: usize },
    /// Fit + persist the mapping; response message carries the rms error.
    CalFinish,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Dictation,
    Gaze,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct State {
    pub dictation: bool,
    pub gaze: bool,
    #[serde(default)]
    pub gaze_calibrated: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    #[serde(default)]
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// State dir: injected by herdr, with a fallback for standalone testing.
pub fn state_dir() -> PathBuf {
    match std::env::var_os("HERDR_PLUGIN_STATE_DIR") {
        Some(d) => PathBuf::from(d),
        None => {
            let home = std::env::var_os("HOME").expect("HOME not set");
            PathBuf::from(home).join(".local/state/herdr-handsfree")
        }
    }
}

pub fn socket_path() -> PathBuf {
    state_dir().join("control.sock")
}

/// Send one request to the daemon and read one response.
pub fn request(req: &Request) -> std::io::Result<Response> {
    let mut stream = UnixStream::connect(socket_path())?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut line = serde_json::to_string(req).expect("serialize request");
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    let mut reader = BufReader::new(stream);
    let mut buf = String::new();
    reader.read_line(&mut buf)?;
    serde_json::from_str(&buf).map_err(std::io::Error::other)
}

/// True when a daemon answers a ping on the control socket.
pub fn daemon_alive() -> bool {
    matches!(request(&Request::Ping), Ok(r) if r.ok)
}
