//! The on-screen controls, built with [egui](https://docs.rs/egui).
//!
//! egui draws as its own pass over the finished scene (see
//! [`crate::gpu::State::render`]), so nothing here touches the world. The panel
//! is a material picker, a brush picker, the tools, a size, and the clock:
//! pause, a single step and a speed. The keyboard
//! shortcuts in [`crate::app`] drive the very same [`Controls`], which is what
//! keeps the two in step.
//!
//! A material and a brush are picked together: the material is what a stroke
//! puts down and the brush is how. A tool is picked instead of them, and does
//! something else with the cursor.

use egui::{Color32, RichText, Stroke};

use crate::materials::{EMPTY, MaterialId, Registry, SAND};

/// The slowest the world can be run, as a multiple of real time. Below a
/// quarter speed sand falls so slowly it looks stuck, and pausing does that job
/// better.
pub const MIN_SPEED: f64 = 0.25;

/// The fastest, as a multiple of real time. Four times is 240 ticks a second,
/// which a laptop GPU still keeps up with; past that the frame rate drops and
/// the world runs no faster anyway.
pub const MAX_SPEED: f64 = 4.0;

/// What a drag does. Most of the time it paints the chosen material with the
/// chosen brush; the wind tool instead blows a gust the way the cursor is
/// swept, without putting any cells down, and a plugin tool does whatever its
/// script says.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Paint [`Controls::material`] with [`Controls::brush`].
    Paint,
    /// Blow wind the way the cursor sweeps.
    Wind,
    /// A tool a plugin registered, by its position in the tool list from
    /// [`crate::plugins::Plugins::names`].
    Plugin(usize),
}

/// The state the panel and the keyboard shortcuts share. egui reads and writes
/// it in place each frame, and [`crate::app`] pokes the same fields from its key
/// handler.
pub struct Controls {
    /// What a drag does: paint, blow wind, or run a plugin tool.
    pub tool: Tool,
    /// The material a stroke paints, when [`Controls::tool`] is
    /// [`Tool::Paint`]. [`EMPTY`] is the eraser.
    pub material: MaterialId,
    /// How a stroke paints, when [`Controls::tool`] is [`Tool::Paint`]: a
    /// position in the brush list from [`crate::plugins::Plugins::names`].
    /// Zero is the plain disk brush, since that is the first one built in.
    pub brush: usize,
    /// The size, in grid cells: the brush radius, or the wind tool's gust
    /// radius, or whatever a plugin makes of it.
    pub radius: i32,
    /// Whether the world is frozen. Painting and the wind tool still work on a
    /// paused world; only the ticks stop.
    pub paused: bool,
    /// How fast the world runs, as a multiple of real time, between
    /// [`MIN_SPEED`] and [`MAX_SPEED`]. One is the usual sixty ticks a second.
    pub speed: f64,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            tool: Tool::Paint,
            material: SAND,
            brush: 0,
            radius: 8,
            paused: false,
            speed: 1.0,
        }
    }
}

impl Controls {
    /// The rate the world should advance at this frame, as a multiple of real
    /// time: the chosen speed, or nothing at all while paused.
    pub fn rate(&self) -> f64 {
        if self.paused { 0.0 } else { self.speed }
    }

    /// Halve the speed, down to [`MIN_SPEED`].
    pub fn slower(&mut self) {
        self.speed = (self.speed / 2.0).max(MIN_SPEED);
    }

    /// Double the speed, up to [`MAX_SPEED`].
    pub fn faster(&mut self) {
        self.speed = (self.speed * 2.0).min(MAX_SPEED);
    }
}

/// What the panel's buttons asked for this frame. The keyboard shortcuts do the
/// same things directly, so this only carries what was clicked.
#[derive(Default)]
pub struct Actions {
    pub clear: bool,
    /// Advance the world by exactly one tick, pausing it first if it was
    /// running.
    pub step: bool,
}

/// Build the panel for this frame and report which buttons were hit.
///
/// `registry` is where the material swatches come from, `brushes` and `tools`
/// the names of the plugin brushes and tools in the order [`Controls::brush`]
/// and [`Tool::Plugin`] count them, and `status` a line for the foot of the
/// panel: what the last plugin drop did, or a hint.
pub fn draw(
    ctx: &egui::Context,
    c: &mut Controls,
    registry: &Registry,
    brushes: &[String],
    tools: &[String],
    status: &str,
) -> Actions {
    let mut actions = Actions::default();
    let button_size = egui::vec2(130.0, 18.0);
    let chosen = Stroke::new(2.0, Color32::WHITE);

    egui::Window::new("Sandy")
        .default_pos([8.0, 8.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label("Material");
            for (id, info) in registry.materials().iter().enumerate() {
                if !info.pickable() {
                    continue;
                }
                let id = id as MaterialId;
                // Air is the eraser, and reads better under that name than under
                // its own.
                let name = if id == EMPTY { "Eraser" } else { info.name };
                let fill = to_color32(info.color);
                let mut button = egui::Button::new(RichText::new(name).color(contrast(fill)))
                    .fill(fill)
                    .min_size(button_size);
                // Only highlight the chosen material while painting is what a
                // drag would actually do.
                if id == c.material && c.tool == Tool::Paint {
                    button = button.stroke(chosen);
                }
                if ui.add(button).clicked() {
                    c.material = id;
                    c.tool = Tool::Paint; // picking a material means painting
                }
            }

            // The brushes go with the materials: one of each is chosen at a
            // time, and picking either means painting.
            ui.separator();
            ui.label("Brush");
            for (index, name) in brushes.iter().enumerate() {
                let mut button = egui::Button::new(name).min_size(button_size);
                if index == c.brush && c.tool == Tool::Paint {
                    button = button.stroke(chosen);
                }
                if ui.add(button).clicked() {
                    c.brush = index;
                    c.tool = Tool::Paint;
                }
            }

            ui.separator();
            ui.label("Tool");
            let mut wind = egui::Button::new("Wind").min_size(button_size);
            if c.tool == Tool::Wind {
                wind = wind.stroke(chosen);
            }
            if ui.add(wind).clicked() {
                c.tool = Tool::Wind;
            }
            for (index, name) in tools.iter().enumerate() {
                let mut button = egui::Button::new(name).min_size(button_size);
                if c.tool == Tool::Plugin(index) {
                    button = button.stroke(chosen);
                }
                if ui.add(button).clicked() {
                    c.tool = Tool::Plugin(index);
                }
            }

            ui.separator();
            let label = match c.tool {
                Tool::Paint => "Brush size",
                Tool::Wind => "Gust size",
                Tool::Plugin(_) => "Size",
            };
            ui.add(egui::Slider::new(&mut c.radius, 1..=60).text(label));

            // Time. The speed slider is logarithmic so that half speed and
            // double speed sit the same distance either side of one.
            ui.separator();
            ui.label("Time");
            let label = if c.paused { "Resume" } else { "Pause" };
            if ui
                .add(egui::Button::new(label).min_size(button_size))
                .clicked()
            {
                c.paused = !c.paused;
            }
            actions.step |= ui
                .add(egui::Button::new("Step one tick").min_size(button_size))
                .clicked();
            ui.add(
                egui::Slider::new(&mut c.speed, MIN_SPEED..=MAX_SPEED)
                    .logarithmic(true)
                    .suffix("x")
                    .text("Speed"),
            );

            ui.separator();
            actions.clear |= ui
                .add(egui::Button::new("Clear").min_size(button_size))
                .clicked();

            ui.separator();
            ui.label(
                RichText::new(
                    "Hold left mouse to draw. Pick Wind and sweep to blow a gust. \
                     Space pauses, . steps a tick, - and = change the speed.",
                )
                .small()
                .weak(),
            );
            ui.label(RichText::new(status).small().weak());
        });

    actions
}

/// A material's swatch colour as an egui [`Color32`].
fn to_color32(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

/// Black or white text, whichever reads better over a swatch. Rec. 601 luma.
fn contrast(c: Color32) -> Color32 {
    let luma = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
    if luma > 140.0 {
        Color32::BLACK
    } else {
        Color32::WHITE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_halves_and_doubles_within_bounds() {
        let mut c = Controls::default();
        assert_eq!(c.rate(), 1.0);
        c.faster();
        assert_eq!(c.speed, 2.0);
        for _ in 0..10 {
            c.faster();
        }
        assert_eq!(c.speed, MAX_SPEED);
        for _ in 0..10 {
            c.slower();
        }
        assert_eq!(c.speed, MIN_SPEED);
    }

    #[test]
    fn pausing_stops_the_clock_and_keeps_the_speed() {
        let mut c = Controls::default();
        c.faster();
        c.paused = true;
        assert_eq!(c.rate(), 0.0);
        c.paused = false;
        assert_eq!(c.rate(), 2.0);
    }
}
