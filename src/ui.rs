//! The on-screen controls, built with [egui](https://docs.rs/egui).
//!
//! egui draws as its own pass over the finished scene (see
//! [`crate::gpu::State::render`]), so nothing here touches the world. The panel
//! is a material picker, the tools, and a brush size. The keyboard shortcuts in
//! [`crate::app`] drive the very same [`Controls`], which is what keeps the two
//! in step.

use egui::{Color32, RichText, Stroke};

use crate::materials::{EMPTY, MaterialId, Registry, SAND};

/// What a drag does. Most of the time it paints the chosen material; the wind
/// tool instead blows a gust the way the cursor is swept, without putting any
/// cells down, and a plugin tool does whatever its script says.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Paint the chosen [`Controls::material`].
    Paint,
    /// Blow wind the way the cursor sweeps.
    Wind,
    /// A tool a plugin registered, by its position in
    /// [`crate::plugins::Plugins::tool_names`].
    Plugin(usize),
}

/// The state the panel and the keyboard shortcuts share. egui reads and writes
/// it in place each frame, and [`crate::app`] pokes the same fields from its key
/// handler.
pub struct Controls {
    /// What a drag does: paint, or blow wind.
    pub tool: Tool,
    /// The material the brush paints with, when [`Controls::tool`] is
    /// [`Tool::Paint`]. [`EMPTY`] is the eraser.
    pub material: MaterialId,
    /// Brush radius, in grid cells. Doubles as the wind tool's gust radius.
    pub brush: i32,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            tool: Tool::Paint,
            material: SAND,
            brush: 8,
        }
    }
}

/// What the panel's buttons asked for this frame. The keyboard shortcuts do the
/// same things directly, so this only carries what was clicked.
#[derive(Default)]
pub struct Actions {
    pub clear: bool,
}

/// Build the panel for this frame and report which buttons were hit.
///
/// `registry` is where the material swatches come from, `tools` the names of
/// the plugin tools in the order [`Tool::Plugin`] counts them, and `status` a
/// line for the foot of the panel: what the last plugin drop did, or a hint.
pub fn draw(
    ctx: &egui::Context,
    c: &mut Controls,
    registry: &Registry,
    tools: &[String],
    status: &str,
) -> Actions {
    let mut actions = Actions::default();
    let button_size = egui::vec2(130.0, 18.0);

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
                    button = button.stroke(Stroke::new(2.0, Color32::WHITE));
                }
                if ui.add(button).clicked() {
                    c.material = id;
                    c.tool = Tool::Paint; // picking a material means painting
                }
            }

            ui.separator();
            let mut wind = egui::Button::new("Wind").min_size(button_size);
            if c.tool == Tool::Wind {
                wind = wind.stroke(Stroke::new(2.0, Color32::WHITE));
            }
            if ui.add(wind).clicked() {
                c.tool = Tool::Wind;
            }
            for (index, name) in tools.iter().enumerate() {
                let mut button = egui::Button::new(name).min_size(button_size);
                if c.tool == Tool::Plugin(index) {
                    button = button.stroke(Stroke::new(2.0, Color32::WHITE));
                }
                if ui.add(button).clicked() {
                    c.tool = Tool::Plugin(index);
                }
            }

            ui.separator();
            let label = if c.tool == Tool::Wind {
                "Gust size"
            } else {
                "Brush"
            };
            ui.add(egui::Slider::new(&mut c.brush, 1..=60).text(label));

            ui.separator();
            actions.clear |= ui
                .add(egui::Button::new("Clear").min_size(button_size))
                .clicked();

            ui.separator();
            ui.label(
                RichText::new("Hold left mouse to draw. Pick Wind and sweep to blow a gust.")
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
