# herdr-handsfree

A [herdr](https://herdr.dev) plugin for hands-free control on macOS: local voice dictation typed into the focused pane, and a webcam-gaze-driven mouse cursor. Everything runs on-device — audio and video never leave your machine.

- **Dictation**: microphone → silero VAD → Qwen3-ASR on candle (Metal) → text delivered to the focused herdr pane. Agent panes get a first-class `herdr agent prompt`; plain terminals get literal text.
- **Gaze mouse**: webcam → MediaPipe FaceMesh + Iris (ONNX) → calibrated affine mapping → smoothed cursor moves at camera rate. Move only — no clicks in v1.

## Install

Requires macOS on Apple Silicon. The plugin's build step downloads the prebuilt binary from the GitHub release — no toolchain needed.

```sh
herdr plugin install RanolP/herdr-handsfree
```

To build from source instead (needs only Rust — the default STT backend is pure Rust, no cmake):

```sh
git clone https://github.com/RanolP/herdr-handsfree
cd herdr-handsfree
cargo build --release
herdr plugin link .
```

Models download automatically on first use into the plugin state dir: Qwen3-ASR `1.7B` (~4.5 GB, 52 languages), silero VAD (~2 MB), FaceMesh + Iris (~5 MB). `asr_model = "0.6B"` is the lighter option at ~1.7 GB, at roughly a third more Korean errors.

## macOS permissions

All prompts land on the app hosting herdr (your terminal / the herdr app):

1. **Microphone** — prompted on first dictation toggle.
2. **Camera** — prompted on first gaze toggle / calibration.
3. **Accessibility** (System Settings → Privacy & Security → Accessibility) — required for the gaze mouse to move the cursor; add your terminal app manually if cursor moves silently fail.

## Use

Actions (bind or run via `herdr plugin action invoke`):

- `ranolp.handsfree.toggle-dictation` — start/stop dictation
- `ranolp.handsfree.toggle-gaze` — start/stop the gaze mouse

Panes: `status` (live on/off view), `calibrate` (guided 9-point gaze calibration — the cursor jumps to 9 screen points; look at it and press Enter each time).

Suggested keybindings in your herdr `config.toml`:

```toml
[[keys.command]]
key = "prefix+d"
type = "plugin_action"
command = "ranolp.handsfree.toggle-dictation"
description = "toggle dictation"

[[keys.command]]
key = "prefix+g"
type = "plugin_action"
command = "ranolp.handsfree.toggle-gaze"
description = "toggle gaze mouse"
```

## Config

Optional `config.toml` in the plugin config dir (`herdr plugin config-dir ranolp.handsfree`):

```toml
asr_model = "1.7B"           # 1.7B | 0.6B (smaller, faster, less accurate)
language = "auto"            # or "korean", "english", ... (ISO codes like "ko" also work)
vad = "silero"               # or "energy"
microphone = ""              # substring of the input device name; "" = system default
smoothing_min_cutoff = 1.0   # lower = smoother, laggier cursor
smoothing_beta = 0.01        # higher = snappier fast moves
```

## Honest expectations

- Webcam gaze is coarse: expect ~2–4 cm of on-screen accuracy after calibration. It moves the pointer to a region; it is not pixel-precise. Recalibrate after changing posture, chair height, or display.
- Qwen3-ASR `1.7B` decodes at roughly 0.35–0.45× realtime on Apple Silicon. Dictation transcribes while you talk, so text lands about 1–2 s after you stop speaking rather than after your whole sentence is processed — still dictation-into-prompt latency, not live per-word typing.
- The model loads once per daemon process (~1.6 s warm) and then stays resident, so the first dictation of a session costs ~4.5 GB of RAM until the daemon stops. Set `asr_model = "0.6B"` on a 16 GB machine.
- The whisper.cpp backend is still available as `cargo build --release --features whisper`, which reads `whisper_model` instead of `asr_model`. It needs cmake (`brew install cmake`).
- herdr is pre-1.0; the plugin pins `min_herdr_version = "0.7.5"` and CLI surface drift is expected.

## Development

```sh
cargo test                                  # self-checks: control socket, VAD+STT, mapping fit
herdr plugin log                            # action/startup command output
target/release/herdr-handsfree status       # poke the daemon directly
```

The daemon logs to `daemon.log` in the plugin state dir.
