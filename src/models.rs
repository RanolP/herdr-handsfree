//! First-run model downloads into the plugin state dir.

use std::io::Write;
use std::path::PathBuf;

const WHISPER_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
// MediaPipe FaceMesh + Iris exported to ONNX (Apache-2.0), hosted by the
// ailia-models project.
const FACEMESH_URL: &str = "https://storage.googleapis.com/ailia-models/facemesh/facemesh.onnx";
const IRIS_URL: &str = "https://storage.googleapis.com/ailia-models/mediapipe_iris/iris.onnx";
const SILERO_URL: &str =
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx";

/// Whisper model size, e.g. "tiny", "base", "small". Multilingual variants.
pub fn whisper_model_size() -> String {
    std::env::var("HANDSFREE_MODEL")
        .unwrap_or_else(|_| crate::config::get().whisper_model.clone())
}

/// Return the local path of the whisper ggml model, downloading it first if missing.
pub fn ensure_whisper_model() -> std::io::Result<PathBuf> {
    let size = whisper_model_size();
    download_once(
        &format!("ggml-{size}.bin"),
        &format!("{WHISPER_BASE_URL}/ggml-{size}.bin"),
    )
}

/// Return (facemesh, iris) ONNX model paths, downloading them if missing.
pub fn ensure_gaze_models() -> std::io::Result<(PathBuf, PathBuf)> {
    Ok((
        download_once("facemesh.onnx", FACEMESH_URL)?,
        download_once("iris.onnx", IRIS_URL)?,
    ))
}

pub fn ensure_silero_model() -> std::io::Result<PathBuf> {
    download_once("silero_vad.onnx", SILERO_URL)
}

fn download_once(filename: &str, url: &str) -> std::io::Result<PathBuf> {
    let dir = crate::control::state_dir().join("models");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    if path.exists() {
        return Ok(path);
    }
    eprintln!("downloading {filename} from {url} ...");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| std::io::Error::other(format!("model download failed: {e}")))?;
    let tmp = path.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp.into_reader(), &mut file)?;
    file.flush()?;
    std::fs::rename(&tmp, &path)?;
    eprintln!("model saved to {}", path.display());
    Ok(path)
}
