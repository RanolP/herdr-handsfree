//! M2 self-check: synthesize speech with macOS `say`, run it through the
//! VAD + whisper path, assert a sane transcript. Uses the tiny model
//! (downloaded on first run into the fallback state dir).

use std::process::Command;

#[test]
#[cfg(target_os = "macos")]
fn wav_through_vad_and_whisper() {
    // CI runners synthesize silent/garbled audio with `say` (paravirtual
    // audio stack), so this check only runs locally unless opted in.
    if std::env::var_os("CI").is_some() && std::env::var_os("HANDSFREE_E2E").is_none() {
        eprintln!("skipping: CI without HANDSFREE_E2E");
        return;
    }
    let wav = std::env::temp_dir().join(format!("handsfree-stt-{}.wav", std::process::id()));
    let ok = Command::new("say")
        .args(["-o"])
        .arg(&wav)
        .args(["--data-format=LEI16@16000", "hello world, this is a dictation test"])
        .status()
        .expect("spawn say")
        .success();
    assert!(ok, "say failed");

    let out = Command::new(env!("CARGO_BIN_EXE_herdr-handsfree"))
        .arg("transcribe-file")
        .arg(&wav)
        .env("HANDSFREE_MODEL", "tiny")
        .output()
        .expect("spawn transcribe-file");
    let _ = std::fs::remove_file(&wav);
    let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
    assert!(
        out.status.success() && text.contains("hello world"),
        "unexpected transcript: {text:?}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
