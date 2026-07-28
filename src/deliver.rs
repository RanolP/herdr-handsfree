//! Deliver transcribed text to the focused herdr pane.

use std::process::Command;

fn herdr_bin() -> String {
    std::env::var("HERDR_BIN_PATH").unwrap_or_else(|_| "herdr".to_string())
}

struct FocusedPane {
    pane_id: String,
    is_agent: bool,
}

/// Focused pane at delivery time, from the live pane list (the daemon's own
/// HERDR_PANE_ID env is stale — it names whichever pane started the daemon).
fn focused_pane() -> Result<FocusedPane, String> {
    let out = Command::new(herdr_bin())
        .args(["pane", "list"])
        .output()
        .map_err(|e| format!("herdr pane list: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "herdr pane list failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("pane list json: {e}"))?;
    let panes = json["result"]["panes"]
        .as_array()
        .ok_or("pane list: missing panes array")?;
    let pane = panes
        .iter()
        .find(|p| p["focused"].as_bool() == Some(true))
        .ok_or("no focused pane")?;
    Ok(FocusedPane {
        pane_id: pane["pane_id"]
            .as_str()
            .ok_or("focused pane has no pane_id")?
            .to_string(),
        is_agent: pane["agent"].as_str().is_some(),
    })
}

/// Type `text` into the focused pane: agent panes get a first-class prompt,
/// plain terminals get literal text.
pub fn deliver(text: &str) -> Result<(), String> {
    let pane = focused_pane()?;
    let args: [&str; 4] = if pane.is_agent {
        ["agent", "prompt", &pane.pane_id, text]
    } else {
        ["pane", "send-text", &pane.pane_id, text]
    };
    let out = Command::new(herdr_bin())
        .args(args)
        .output()
        .map_err(|e| format!("herdr {}: {e}", args[..2].join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "herdr {} failed: {}",
            args[..2].join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}
