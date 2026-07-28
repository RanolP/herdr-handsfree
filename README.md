# herdr-handsfree

A [herdr](https://herdr.dev) plugin for hands-free control on macOS: local voice dictation typed into the focused pane, and a webcam-gaze-driven mouse cursor. Everything runs on-device — audio and video never leave your machine.

- **Dictation**: microphone → silero VAD → whisper.cpp (Metal) → text delivered to the focused herdr pane. Agent panes get a first-class `herdr agent prompt`; plain terminals get literal text.
- **Gaze mouse**: webcam → MediaPipe FaceMesh + Iris (ONNX) → calibrated affine mapping → smoothed cursor moves at camera rate. Move only — no clicks in v1.

## Install

Requires macOS on Apple Silicon. The plugin's build step downloads the prebuilt binary from the GitHub release — no toolchain needed.

```sh
herdr plugin install RanolP/herdr-handsfree
```

To build from source instead (needs Rust and cmake, `brew install cmake`):

```sh
git clone https://github.com/RanolP/herdr-handsfree
cd herdr-handsfree
cargo build --release
herdr plugin link .
```

Models download automatically on first use into the plugin state dir: whisper ggml `small` (~466 MB, multilingual), silero VAD (~2 MB), FaceMesh + Iris (~5 MB).

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
whisper_model = "small"      # tiny | base | small | medium | large-v3
language = "auto"            # or "ko", "en", ...
vad = "silero"               # or "energy"
smoothing_min_cutoff = 1.0   # lower = smoother, laggier cursor
smoothing_beta = 0.01        # higher = snappier fast moves
```

## Honest expectations

- Webcam gaze is coarse: expect ~2–4 cm of on-screen accuracy after calibration. It moves the pointer to a region; it is not pixel-precise. Recalibrate after changing posture, chair height, or display.
- Whisper `small` transcribes an utterance in ~0.5–2 s on Apple Silicon — dictation-into-prompt latency, not live per-word typing.
- herdr is pre-1.0; the plugin pins `min_herdr_version = "0.7.5"` and CLI surface drift is expected.

## Development

```sh
cargo test                                  # self-checks: control socket, VAD+whisper, mapping fit
herdr plugin log                            # action/startup command output
target/release/herdr-handsfree status       # poke the daemon directly
```

The daemon logs to `daemon.log` in the plugin state dir.
