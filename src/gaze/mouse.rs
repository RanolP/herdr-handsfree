//! Cursor movement via CGEvent. Needs Accessibility permission for the app
//! hosting the daemon (usually your terminal).

use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use core_graphics::geometry::CGPoint;

/// Main display bounds in points: (origin_x, origin_y, width, height).
pub fn main_display_bounds() -> (f64, f64, f64, f64) {
    let b = CGDisplay::main().bounds();
    (b.origin.x, b.origin.y, b.size.width, b.size.height)
}

pub fn move_cursor(x: f64, y: f64) -> Result<(), String> {
    let (ox, oy, w, h) = main_display_bounds();
    let x = x.clamp(ox, ox + w - 1.0);
    let y = y.clamp(oy, oy + h - 1.0);
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| "CGEventSource failed".to_string())?;
    let event = CGEvent::new_mouse_event(
        source,
        CGEventType::MouseMoved,
        CGPoint::new(x, y),
        CGMouseButton::Left,
    )
    .map_err(|_| {
        "CGEvent create failed — grant Accessibility permission to your terminal app".to_string()
    })?;
    event.post(CGEventTapLocation::HID);
    Ok(())
}
