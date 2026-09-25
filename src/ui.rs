//! The on-screen controls, built with [egui](https://docs.rs/egui).
//!
//! egui draws as its own pass over the finished scene (see
//! [`crate::gpu::State::render`]), so nothing here touches the world. The panel
//! is a material picker, a brush picker, the tools, a size, the clock (pause,
//! a single step and a speed), the world: a preset, a seed, and the buttons
//! that build one, and capture: a screenshot and a recording, each with a
//! format. The keyboard shortcuts in [`crate::app`] drive the very same
//! [`Controls`], which is what keeps the two in step.
//!
//! A material and a brush are picked together: the material is what a stroke
//! puts down and the brush is how. A tool is picked instead of them, and does
//! something else with the cursor.

use std::time::Duration;

use egui::{Color32, RichText, Stroke};

use crate::capture::{AnimationFormat, ImageFormat};
use crate::materials::{EMPTY, MaterialId, Registry, SAND};
use crate::view::{MAX_ZOOM, MIN_ZOOM, View};

/// The most digits the seed box takes. Nine digits always fit a `u32`, and
/// that is more seeds than anyone will type.
const MAX_SEED_DIGITS: usize = 9;

/// A rolled seed is kept below this so it always fits the seed box.
const SEED_RANGE: u32 = 1_000_000_000;

/// A fresh seed, from a random number generator that is quick rather than
/// secure, which is all a world seed needs. The game opens on one of these
/// so it does not open on the same world every time, and the Random button
/// rolls another.
pub fn random_seed() -> u32 {
    fastrand::u32(..SEED_RANGE)
}

/// The smallest and largest brush the size slider offers, in cells. The
/// desktop world is three thousand cells across, so the largest is a
/// twentieth of it: enough to pour a lake or wall off a valley in a stroke or
/// two. A smaller world gets a smaller largest brush; see [`max_radius`].
pub const MIN_RADIUS: i32 = 1;
pub const MAX_RADIUS: i32 = 150;

/// The largest brush for a world `width` cells across: a twentieth of it, as
/// [`MAX_RADIUS`] is of the desktop grid, and never less than the smallest.
pub fn max_radius(width: u32) -> i32 {
    (width as i32 / 20).max(MIN_RADIUS)
}

/// How the panel is laid out: for a mouse, or for a finger.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    /// A touchscreen: buttons tall enough to land a thumb on, a panel that
    /// opens folded away so it does not cover a small screen and scrolls
    /// rather than run off the bottom of one, and no talk of keys.
    pub touch: bool,
    /// How far below the top of the window the panel opens, in points, to
    /// keep clear of a status bar or a notch.
    pub top_inset: f32,
}

impl Layout {
    /// A desktop window: a mouse, and nothing in the way at the top.
    pub const DESKTOP: Layout = Layout {
        touch: false,
        top_inset: 0.0,
    };
}

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
    /// The largest the size slider goes to, from [`max_radius`] for the
    /// world's width. [`MAX_RADIUS`] until the world's size is known.
    pub max_radius: i32,
    /// Whether the world is frozen. Painting and the wind tool still work on a
    /// paused world; only the ticks stop.
    pub paused: bool,
    /// How fast the world runs, as a multiple of real time, between
    /// [`MIN_SPEED`] and [`MAX_SPEED`]. One is the usual sixty ticks a second.
    pub speed: f64,
    /// The world preset the next generation builds: a position in the world
    /// list from [`crate::plugins::Plugins::world_names`].
    pub world: usize,
    /// The world seed, kept as text so it can be typed into a box. It is
    /// parsed when a world is actually built; see [`Controls::seed_value`].
    pub seed: String,
    /// What a screenshot is saved as.
    pub screenshot_format: ImageFormat,
    /// What a recording is saved as. Read when the recording starts.
    pub recording_format: AnimationFormat,
    /// The zoom, and which part of the world the window shows. The wheel,
    /// a pinch and the keys move it as the panel's slider does.
    pub view: View,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            tool: Tool::Paint,
            material: SAND,
            brush: 0,
            radius: 8,
            max_radius: MAX_RADIUS,
            paused: false,
            speed: 1.0,
            world: 0,
            seed: random_seed().to_string(),
            screenshot_format: ImageFormat::default(),
            recording_format: AnimationFormat::default(),
            view: View::default(),
        }
    }
}

impl Controls {
    /// The seed as a number: zero if the box is empty or holds anything that
    /// is not a `u32`.
    pub fn seed_value(&self) -> u32 {
        self.seed.trim().parse().unwrap_or(0)
    }

    /// Put a seed in the box, as the Random button does.
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed.to_string();
    }

    /// Fit the size slider to a world `width` cells across, and the size to
    /// the slider.
    pub fn fit_to_width(&mut self, width: u32) {
        self.max_radius = max_radius(width);
        self.radius = self.radius.clamp(MIN_RADIUS, self.max_radius);
    }

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
    /// Build the chosen world from the seed in the box.
    pub generate: bool,
    /// Roll a fresh seed into the box and build the chosen world from it.
    pub randomize: bool,
    /// Save the next frame as a screenshot.
    pub screenshot: bool,
    /// Start a recording, or stop the one running.
    pub record: bool,
    /// Where the panel was drawn, in points: its title bar when it is
    /// folded away, the whole of it when open. A press inside it is the
    /// panel's, however egui feels about the pointer at that moment.
    pub panel: Option<egui::Rect>,
}

/// The names of what the plugins have registered, in the order
/// [`Controls::brush`], [`Tool::Plugin`] and [`Controls::world`] count them.
pub struct Names {
    pub brushes: Vec<String>,
    pub tools: Vec<String>,
    pub worlds: Vec<String>,
}

/// Build the panel for this frame and report which buttons were hit.
///
/// `registry` is where the material swatches come from, `names` the plugin
/// brushes, tools and worlds, `recording` how long the recording has been
/// running if one is, `status` a line for the foot of the panel: what the
/// last plugin drop or capture did, or a hint, and `layout` whether it is
/// being used with a mouse or a finger.
pub fn draw(
    ctx: &egui::Context,
    c: &mut Controls,
    registry: &Registry,
    names: &Names,
    recording: Option<Duration>,
    status: &str,
    layout: Layout,
) -> Actions {
    let Names {
        brushes,
        tools,
        worlds,
    } = names;
    let mut actions = Actions::default();
    let (button_size, action_size) = if layout.touch {
        (egui::vec2(150.0, 30.0), egui::vec2(90.0, 30.0))
    } else {
        (egui::vec2(130.0, 18.0), egui::vec2(78.0, 18.0))
    };
    let chosen = Stroke::new(2.0, Color32::WHITE);

    let mut window = egui::Window::new("Sandy")
        .default_pos([8.0, 8.0 + layout.top_inset])
        .resizable(false);
    if layout.touch {
        // A phone's screen is small, so the panel opens folded away to its
        // title and scrolls once open, rather than covering the world or
        // running off the bottom.
        let room = ctx.content_rect().height() - layout.top_inset - 24.0;
        window = window
            .default_open(false)
            .vscroll(true)
            .max_height(room.max(120.0));
    }
    let shown = window.show(ctx, |ui| {
        if layout.touch {
            // Room to land a finger on: the sliders and boxes as tall as
            // the buttons, and a little more air between rows.
            let spacing = ui.spacing_mut();
            spacing.interact_size.y = button_size.y;
            spacing.item_spacing.y = 6.0;
            spacing.slider_width = button_size.x;
        }
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
        ui.add(egui::Slider::new(&mut c.radius, MIN_RADIUS..=c.max_radius).text(label));

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

        // The view: how far in, about the middle of the window, and a way
        // back out to the whole world. The slider is logarithmic too, so
        // doubling is the same distance wherever it starts.
        ui.separator();
        ui.label("View");
        let mut zoom = c.view.zoom();
        if ui
            .add(
                egui::Slider::new(&mut zoom, MIN_ZOOM..=MAX_ZOOM)
                    .logarithmic(true)
                    .suffix("x")
                    .text("Zoom"),
            )
            .changed()
        {
            c.view.set_zoom(zoom);
        }
        if ui
            .add_enabled(
                !c.view.is_whole(),
                egui::Button::new("Fit the world").min_size(button_size),
            )
            .clicked()
        {
            c.view.reset();
        }

        // The world: which landscape, from which seed. Picking another
        // landscape builds it there and then, so the change can be seen
        // without a second click; so does pressing Enter in the seed box.
        ui.separator();
        ui.label("World");
        if worlds.is_empty() {
            ui.label(RichText::new("No world plugins loaded.").small().weak());
        } else {
            c.world = c.world.min(worlds.len() - 1);
            let before = c.world;
            egui::ComboBox::from_id_salt("world")
                .width(button_size.x)
                .selected_text(&worlds[c.world])
                .show_ui(ui, |ui| {
                    for (index, name) in worlds.iter().enumerate() {
                        ui.selectable_value(&mut c.world, index, name);
                    }
                });
            actions.generate |= c.world != before;
        }
        ui.horizontal(|ui| {
            ui.label("Seed");
            let response = ui.add(
                egui::TextEdit::singleline(&mut c.seed)
                    .char_limit(MAX_SEED_DIGITS)
                    .desired_width(button_size.x - 40.0),
            );
            actions.generate |=
                response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        });
        // Digits only, so whatever is in the box parses as a seed.
        c.seed.retain(|ch| ch.is_ascii_digit());
        ui.horizontal(|ui| {
            actions.generate |= ui.button("Generate").clicked();
            actions.randomize |= ui.button("Random").clicked();
            actions.clear |= ui.button("Clear").clicked();
        });

        // Capture: a screenshot of the next frame, or a recording until
        // the button is pressed again, each in the format beside it. The
        // recording's format is fixed once it has started, so the box
        // is greyed out until it stops.
        ui.separator();
        ui.label("Capture");
        ui.horizontal(|ui| {
            actions.screenshot |= ui
                .add(egui::Button::new("Screenshot").min_size(action_size))
                .clicked();
            egui::ComboBox::from_id_salt("screenshot format")
                .width(button_size.x - action_size.x - 8.0)
                .selected_text(c.screenshot_format.name())
                .show_ui(ui, |ui| {
                    for format in ImageFormat::ALL {
                        ui.selectable_value(&mut c.screenshot_format, format, format.name());
                    }
                });
        });
        ui.horizontal(|ui| {
            let label = match recording {
                Some(t) => format!("Stop {:.1} s", t.as_secs_f64()),
                None => "Record".to_string(),
            };
            actions.record |= ui
                .add(egui::Button::new(label).min_size(action_size))
                .clicked();
            ui.add_enabled_ui(recording.is_none(), |ui| {
                egui::ComboBox::from_id_salt("recording format")
                    .width(button_size.x - action_size.x - 8.0)
                    .selected_text(c.recording_format.name())
                    .show_ui(ui, |ui| {
                        for format in AnimationFormat::ALL {
                            ui.selectable_value(&mut c.recording_format, format, format.name());
                        }
                    });
            });
        });

        ui.separator();
        let help = if layout.touch {
            "Drag a finger to draw. Pick Wind and sweep to blow a gust. \
                 Pinch to zoom, and drag two fingers to look around."
        } else {
            "Hold left mouse to draw. Pick Wind and sweep to blow a gust. \
                 Scroll or pinch to zoom, drag with the right button to look around, \
                 F fits the world again. \
                 Space pauses, . steps a tick, - and = change the speed. \
                 G builds the world again, R from a new seed. \
                 S takes a screenshot, V starts and stops a recording."
        };
        ui.label(RichText::new(help).small().weak());
        ui.label(RichText::new(status).small().weak());
    });
    actions.panel = shown.map(|shown| shown.response.rect);

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
    fn the_seed_box_parses_to_a_number_or_to_zero() {
        let mut c = Controls::default();
        assert!(
            c.seed.len() <= MAX_SEED_DIGITS && c.seed.parse::<u32>().is_ok(),
            "the box opens on a seed that fits it: {:?}",
            c.seed
        );
        c.set_seed(42);
        assert_eq!(c.seed, "42");
        assert_eq!(c.seed_value(), 42);
        c.seed = " 7 ".to_string();
        assert_eq!(c.seed_value(), 7);
        c.seed = String::new();
        assert_eq!(c.seed_value(), 0);
        c.seed = "99999999999".to_string();
        assert_eq!(c.seed_value(), 0, "too big for a u32");
    }

    #[test]
    fn rolled_seeds_fit_the_box_and_are_not_all_the_same() {
        let seeds: Vec<u32> = (0..20).map(|_| random_seed()).collect();
        assert!(seeds.iter().all(|&s| s < SEED_RANGE));
        assert!(
            seeds.iter().any(|&s| s != seeds[0]),
            "twenty rolls all came up {}",
            seeds[0]
        );
    }

    #[test]
    fn the_largest_brush_follows_the_width_of_the_world() {
        assert_eq!(max_radius(3000), MAX_RADIUS);
        assert_eq!(max_radius(520), 26);
        assert_eq!(max_radius(10), MIN_RADIUS, "never below the smallest brush");
        let mut c = Controls {
            radius: 100,
            ..Controls::default()
        };
        c.fit_to_width(520);
        assert_eq!(c.max_radius, 26);
        assert_eq!(c.radius, 26, "a brush too big for the world is brought in");
        c.fit_to_width(3000);
        assert_eq!(c.radius, 26, "and one that fits is left alone");
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
