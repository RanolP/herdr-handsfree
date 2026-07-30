//! M2 self-check: synthesize speech with macOS `say`, run it through the
//! VAD + STT path, assert a sane transcript. Uses the smallest model of
//! whichever backend is compiled in (downloaded on first run into the
//! fallback state dir).

use std::process::Command;

#[test]
#[cfg(target_os = "macos")]
fn wav_through_vad_and_stt() {
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
        .args([
            "--data-format=LEI16@16000",
            "hello world, this is a dictation test",
        ])
        .status()
        .expect("spawn say")
        .success();
    assert!(ok, "say failed");

    let out = Command::new(env!("CARGO_BIN_EXE_herdr-handsfree"))
        .arg("transcribe-file")
        .arg(&wav)
        .env("HANDSFREE_ASR_MODEL", "0.6B")
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

/// Live dictation decodes incrementally with prefix rollback, which is not
/// guaranteed to land on the same text as a single batch pass. Korean is where
/// a divergence would hurt most, so gate on it: if these two ever differ, the
/// streaming path is quietly typing worse text than `transcribe-file` shows.
#[test]
#[cfg(target_os = "macos")]
fn streaming_matches_batch_on_korean() {
    if std::env::var_os("CI").is_some() && std::env::var_os("HANDSFREE_E2E").is_none() {
        eprintln!("skipping: CI without HANDSFREE_E2E");
        return;
    }
    let wav = std::env::temp_dir().join(format!("handsfree-ko-{}.wav", std::process::id()));
    let ok = Command::new("say")
        .args(["-v", "Yuna", "-o"])
        .arg(&wav)
        .args([
            "--data-format=LEI16@16000",
            "안녕하세요, 받아쓰기 테스트입니다. 오늘 날씨가 참 좋네요.",
        ])
        .status()
        .expect("spawn say")
        .success();
    assert!(ok, "say -v Yuna failed (voice not installed?)");

    // Force `language = "ko"`: that is the path where qwen3-asr leaves the
    // literal `<asr_text>` separator in the text, and it keeps the comparison
    // independent of whatever the developer's own config.toml says.
    let cfg = std::env::temp_dir().join(format!("handsfree-ko-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&cfg).expect("create config dir");
    std::fs::write(cfg.join("config.toml"), "language = \"ko\"\n").expect("write config");

    let run = |stream: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_herdr-handsfree"));
        cmd.arg("transcribe-file");
        if stream {
            cmd.arg("--stream");
        }
        let out = cmd
            .arg(&wav)
            .env("HERDR_PLUGIN_CONFIG_DIR", &cfg)
            .env("HANDSFREE_ASR_MODEL", "0.6B")
            .env("HANDSFREE_MODEL", "tiny")
            .output()
            .expect("spawn transcribe-file");
        assert!(
            out.status.success(),
            "transcribe-file --stream={stream} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let batch = run(false);
    let streamed = run(true);
    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_dir_all(&cfg);

    assert!(!batch.is_empty(), "batch transcript was empty");
    assert_eq!(streamed, batch, "streaming diverged from batch");
    // The forced-language separator must be stripped on the streaming path too.
    assert!(
        !streamed.contains("<asr_text>"),
        "separator leaked: {streamed:?}"
    );
}
