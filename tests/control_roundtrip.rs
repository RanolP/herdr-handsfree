//! Smallest check that breaks if the daemon control protocol breaks:
//! spawn the real binary with an isolated state dir, toggle, verify, stop.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_herdr-handsfree")
}

fn run(state_dir: &std::path::Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(bin())
        .args(args)
        .env("HERDR_PLUGIN_STATE_DIR", state_dir)
        .output()
        .expect("spawn");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn toggle_roundtrip() {
    let dir = std::env::temp_dir().join(format!("handsfree-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let (ok, _) = run(&dir, &["ensure-daemon"]);
    assert!(ok, "ensure-daemon failed");

    let (ok, out) = run(&dir, &["toggle", "dictation"]);
    assert!(ok && out.contains("dictation=ON"), "unexpected: {out}");

    let (ok, out) = run(&dir, &["status"]);
    assert!(ok && out.contains("dictation: ON") && out.contains("gaze: off"), "unexpected: {out}");

    let (ok, _) = run(&dir, &["stop"]);
    assert!(ok, "stop failed");

    let _ = std::fs::remove_dir_all(&dir);
}
