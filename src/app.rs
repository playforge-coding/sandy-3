//! The window, the input, and the event loop: the glue between winit, the GPU
//! state, egui and the simulation.
//!
//! Window events go to egui first. Whatever it does not want, which is clicks
//! outside the panel and keys pressed with nothing focused, drives the brush and
//! the shortcuts. Each redraw runs egui, steps the world, and hands the
//! tessellated panel to [`State::render`] to layer over the scene.
//!
//! A file dropped on the window is taken to be a plugin (see
//! [`crate::plugins`]) and loaded on the spot. The plugins built into the
//! binary, and then any in [`PLUGIN_DIR`], are loaded before the window opens,
//! and the first world they registered is built from a freshly rolled seed
//! so the game opens on a landscape rather than a blank grid, and a different
//! one each time.
//!
//! Screenshots and recordings go through [`crate::capture`]: each frame, it
//! is asked whether the scene should be copied out, and the frames that come
//! back are handed to it to save.
//!
//! A control script given on the command line (see [`crate::scripting`]) is
//! started once the world is built and advanced at the top of every frame,
//! so what it did shows in that frame, and a frame it asked to let go by is
//! a real one.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::capture::Capture;
use crate::gpu::State;
use crate::materials::{EMPTY, MaterialId};
use crate::plugins::{Kind, PLUGIN_DIR, Plugins, Stroke};
use crate::scripting::{Host, Script, Status};
use crate::sim::Simulation;
use crate::ui;

/// The window size the app opens at, in logical pixels. Twice as wide as it is
/// tall, matching the grid, so nothing is stretched out of shape at the start.
const WINDOW_SIZE: (f64, f64) = (1100.0, 620.0);

/// What the foot of the panel says when there is nothing more recent to say.
const DROP_HINT: &str = "Drop a .js file on the window to load a plugin.";

/// Wind added, in cells per tick, per grid cell the cursor sweeps in a frame.
/// One would have the air move exactly with the cursor; a little more than that
/// makes up for the gust's soft edge, so an ordinary sweep is a proper gust and
/// a brisk flick saturates the field at [`crate::kernels::WIND_MAX`].
const WIND_DRAG_GAIN: f32 = 1.5;

/// How much wider the gust is than the brush. A gust is a soft blob of moving
/// air that fades to nothing at its rim, and the world is a thousand cells
/// across, so one the size of the paint brush would be a pinprick that dies
/// before it has moved anything.
const GUST_SCALE: i32 = 3;

/// The smallest gust the wind tool blows, whatever the brush is set to, so even
/// a fine brush moves something when it is used as a fan.
const MIN_GUST_RADIUS: i32 = 30;

/// One frame of the wind tool: blow a gust the way the cursor has swept, from
/// `from` to `to` since the last frame, with the brush at `radius`. A script
/// sweeping the tool goes through the same function as the mouse.
pub(crate) fn wind_tool(sim: &mut Simulation, from: (i32, i32), to: (i32, i32), radius: i32) {
    let dvx = (to.0 - from.0) as f32 * WIND_DRAG_GAIN;
    let dvy = (to.1 - from.1) as f32 * WIND_DRAG_GAIN;
    let gust = (radius * GUST_SCALE).max(MIN_GUST_RADIUS);
    sim.add_wind_disk(to.0, to.1, gust, dvx, dvy);
}

/// Where the mouse is and what it is doing, plus the [`ui::Controls`] the panel
/// and the shortcuts share.
struct Input {
    cursor: (f64, f64),
    drawing: bool,
    /// The cell the cursor was over on the previous frame of this stroke, so a
    /// drag yields a direction for the wind tool and for plugin tools. `None`
    /// at the start of a stroke, so the first frame only notes where it began.
    last_cell: Option<(i32, i32)>,
    /// The step key was pressed since the last frame: advance one tick, as the
    /// panel's Step button does. Cleared once it has been spent.
    step: bool,
    controls: ui::Controls,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            cursor: (0.0, 0.0),
            drawing: false,
            last_cell: None,
            step: false,
            controls: ui::Controls::default(),
        }
    }
}

struct App {
    state: Option<State>,
    input: Input,
    /// The script side: the materials and tools plugins have added, and the
    /// engine they run in.
    plugins: Plugins,
    /// Screenshots and recordings: what has been asked for, and the threads
    /// writing the files.
    capture: Capture,
    /// The line at the foot of the panel: what the last plugin drop or
    /// capture did.
    status: String,
    /// egui's context: fonts, memory and layout. Cheap to clone, since it is an
    /// `Arc` inside.
    egui_ctx: egui::Context,
    /// Per-window input translation for egui, built once the window exists.
    egui_state: Option<egui_winit::State>,
    /// A control script from the command line, as its name and its text,
    /// waiting for the world to exist so it can be started.
    script_text: Option<(String, String)>,
    /// The control script running, if one is.
    script: Option<Script>,
    /// The script asked for the window to close.
    quit: bool,
}

impl App {
    /// An app that will run `script`, a name and the text, once its window
    /// and world are up.
    fn new(script: Option<(String, String)>) -> Self {
        Self {
            state: None,
            input: Input::default(),
            plugins: Plugins::new(),
            capture: Capture::default(),
            status: DROP_HINT.to_string(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            script_text: script,
            script: None,
            quit: false,
        }
    }

    /// Build the bridge between egui and winit, once the GPU state and so the
    /// window exist. Doing nothing if it is already built, or if it cannot be
    /// built yet, keeps this safe to call more than once.
    fn ensure_egui(&mut self) {
        if self.egui_state.is_some() {
            return;
        }
        let Some(state) = &self.state else {
            return;
        };
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            state.window(),
            Some(state.window().scale_factor() as f32),
            None,
            Some(state.max_texture_side()),
        ));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // `resumed` can fire more than once on some platforms.
        if self.state.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Sandy 3")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_SIZE.0, WINDOW_SIZE.1));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        // The plugins go first, so the world is built knowing their materials
        // rather than having the tables swapped out a moment later. The
        // built-in ones come before the user's, so a user's script can retune
        // a built-in one by name.
        self.plugins.load_builtin();
        self.load_plugin_dir();
        self.state = Some(pollster::block_on(State::new(
            window,
            &self.plugins.registry(),
        )));
        self.apply_commands();
        self.ensure_egui();
        // Open on a landscape, as long as some plugin has provided one.
        if !self.plugins.world_names().is_empty() {
            self.generate();
        }
        // The script goes last, so it finds the world it would see on the
        // screen. Its first requests are answered on the first frame.
        if let Some((name, source)) = self.script_text.take() {
            match Script::start(&self.plugins, &name, &source) {
                Ok(script) => {
                    log::info!("running script {name}");
                    self.status = format!("Running {name}.");
                    self.script = Some(script);
                }
                Err(err) => {
                    log::error!("script {name} failed: {err}");
                    self.status = err;
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Offer the event to egui first. Consumed means it landed on the panel
        // and should not also paint or fire a shortcut.
        let consumed = match (&self.state, &mut self.egui_state) {
            (Some(state), Some(egui_state)) => {
                egui_state.on_window_event(state.window(), &event).consumed
            }
            _ => false,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.resize(size.width, size.height);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor = (position.x, position.y);
            }

            WindowEvent::MouseInput {
                state: button,
                button: MouseButton::Left,
                ..
            } => {
                if button == ElementState::Pressed {
                    self.input.drawing = !consumed;
                    self.input.last_cell = None; // a fresh stroke has no direction yet
                } else {
                    self.input.drawing = false;
                }
            }

            // A file dropped on the window is a plugin. While one is being
            // dragged over, say so, in place of the standing hint.
            WindowEvent::DroppedFile(path) => self.load_plugin(&path),
            WindowEvent::HoveredFile(_) => self.status = "Drop it to load the plugin.".to_string(),
            WindowEvent::HoveredFileCancelled => self.status = DROP_HINT.to_string(),

            // A touchscreen draws exactly as the mouse does. The position rides
            // on the event itself rather than arriving separately, so the cursor
            // is updated here too.
            WindowEvent::Touch(touch) => {
                self.input.cursor = (touch.location.x, touch.location.y);
                match touch.phase {
                    TouchPhase::Started => {
                        self.input.drawing = !consumed;
                        self.input.last_cell = None;
                    }
                    TouchPhase::Moved => {} // keep drawing; the cursor is updated
                    TouchPhase::Ended | TouchPhase::Cancelled => self.input.drawing = false,
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // Skip the shortcuts while egui wants the key.
                if !consumed
                    && event.state == ElementState::Pressed
                    && let PhysicalKey::Code(code) = event.physical_key
                {
                    self.handle_key(code);
                }
            }

            WindowEvent::RedrawRequested => {
                self.redraw();
                if self.quit {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Keep the world moving: ask for the next frame straight away.
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }
}

impl App {
    /// One frame: use the tool under the cursor, run egui, step the world, and
    /// draw the scene with the panel on top.
    fn redraw(&mut self) {
        if self.state.is_none() || self.egui_state.is_none() {
            return;
        }

        // The script first, so whatever it does is in this frame, and a
        // frame it lets go by is the whole of one.
        self.advance_script();

        if self.input.drawing {
            let c = &self.input.controls;
            let (tool, material, brush, radius) = (c.tool, c.material, c.brush, c.radius);
            let (gx, gy) = self
                .state
                .as_ref()
                .unwrap()
                .cursor_to_grid(self.input.cursor);
            let last = self.input.last_cell;
            self.input.last_cell = Some((gx, gy));

            // The wind tool is the one thing a stroke does in Rust. Everything
            // else, the plain brush included, is a script: painting is the
            // chosen brush, and a plugin tool is itself.
            let script = match tool {
                ui::Tool::Paint => Some((Kind::Brush, brush)),
                ui::Tool::Plugin(index) => Some((Kind::Tool, index)),
                ui::Tool::Wind => {
                    // Blow a gust the way the cursor has swept since the last
                    // frame. The first frame of a stroke only notes where it is.
                    if let Some(from) = last {
                        let state = self.state.as_mut().unwrap();
                        wind_tool(&mut state.sim, from, (gx, gy), radius);
                    }
                    None
                }
            };
            if let Some((kind, index)) = script {
                // The script gets the frame and queues what it wants done. One
                // that fails ends the stroke, so a mistake in it shows on the
                // panel once rather than sixty times a second.
                let (px, py) = last.unwrap_or((gx, gy));
                let stroke = Stroke {
                    x: gx,
                    y: gy,
                    px,
                    py,
                    first: last.is_none(),
                    radius,
                    material,
                };
                if let Err(err) = self.plugins.run(kind, index, stroke) {
                    log::error!("plugin failed: {err}");
                    self.status = err;
                    self.input.drawing = false;
                }
                self.apply_commands();
            }
        }

        // Run egui for this frame. The window handle is cloned so it does not
        // hold a borrow of `self.state` while that is mutated below.
        let window = self.state.as_ref().unwrap().window_arc();
        let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(&window);
        let ctx = self.egui_ctx.clone();
        let mut actions = ui::Actions::default();
        let names = ui::Names {
            brushes: self.plugins.names(Kind::Brush),
            tools: self.plugins.names(Kind::Tool),
            worlds: self.plugins.world_names(),
        };
        let now = Instant::now();
        let recording = self.capture.recording_for(now);
        let full_output = ctx.run_ui(raw_input, |ui| {
            actions = ui::draw(
                ui.ctx(),
                &mut self.input.controls,
                &self.plugins.registry(),
                &names,
                recording,
                &self.status,
            );
        });
        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output);
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        if actions.randomize {
            self.randomize();
        } else if actions.generate {
            self.generate();
        }
        if actions.screenshot {
            self.screenshot();
        }
        if actions.record {
            self.toggle_recording();
        }

        if let Some(state) = &mut self.state {
            if actions.clear {
                state.sim.clear();
            }
            // A step is one tick and then stillness, so the world is paused
            // first if it was running; `update` then adds nothing on top.
            if actions.step || std::mem::take(&mut self.input.step) {
                self.input.controls.paused = true;
                state.sim.step();
            }
            state.update(self.input.controls.rate());

            // Copy this frame out if a screenshot or the recording wants it,
            // and pass on whichever earlier ones have finished copying.
            let wanted = self.capture.plan(now);
            let captured = state.render(
                paint_jobs,
                full_output.textures_delta,
                full_output.pixels_per_point,
                wanted.is_some(),
            );
            if let (true, Some(wanted)) = (captured, wanted) {
                self.capture.taken(wanted);
            }
            for frame in state.renderer.take_frames() {
                self.capture.deliver(frame);
            }
        }
        for line in self.capture.reports() {
            log::info!("{line}");
            self.status = line;
        }
    }

    /// Give the control script its turn: answer what it asks until it is done
    /// with this frame, or with everything.
    fn advance_script(&mut self) {
        let (Some(script), Some(state)) = (&mut self.script, &mut self.state) else {
            return;
        };
        let mut host = Host {
            plugins: &mut self.plugins,
            sim: &mut state.sim,
            renderer: &mut state.renderer,
            capture: &mut self.capture,
            controls: &mut self.input.controls,
            clock: None,
        };
        match script.advance(&mut host) {
            Status::Running => {}
            Status::Finished => {
                log::info!("script finished");
                self.status = format!("Script finished. {DROP_HINT}");
                self.script = None;
            }
            Status::Quit => {
                log::info!("script asked to quit");
                self.script = None;
                self.quit = true;
            }
            Status::Failed(err) => {
                log::error!("script failed: {err}");
                self.status = err;
                self.script = None;
            }
        }
    }

    /// Save the next frame as a screenshot, in the format on the panel.
    fn screenshot(&mut self) {
        self.capture
            .screenshot(self.input.controls.screenshot_format);
    }

    /// Start a recording in the format on the panel, or stop the one running.
    fn toggle_recording(&mut self) {
        let format = self.input.controls.recording_format;
        self.status = self.capture.toggle_recording(format, Instant::now());
        log::info!("{}", self.status);
    }

    fn handle_key(&mut self, code: KeyCode) {
        // The number keys pick a material by id, which is the order the picker
        // lists them in: the built-in five, then whatever the plugins added.
        // Zero is air, the eraser. Choosing a material means painting with it.
        let digit: Option<MaterialId> = match code {
            KeyCode::Digit0 | KeyCode::Backspace => Some(EMPTY),
            KeyCode::Digit1 => Some(1),
            KeyCode::Digit2 => Some(2),
            KeyCode::Digit3 => Some(3),
            KeyCode::Digit4 => Some(4),
            KeyCode::Digit5 => Some(5),
            KeyCode::Digit6 => Some(6),
            KeyCode::Digit7 => Some(7),
            KeyCode::Digit8 => Some(8),
            KeyCode::Digit9 => Some(9),
            _ => None,
        };
        if let Some(id) = digit {
            let registry = self.plugins.registry();
            if let Some(info) = registry.materials().get(id as usize)
                && info.pickable()
            {
                self.input.controls.material = id;
                self.input.controls.tool = ui::Tool::Paint;
            }
            return;
        }

        let c = &mut self.input.controls;
        match code {
            // The wind tool: sweep the cursor to blow a gust.
            KeyCode::KeyW => c.tool = ui::Tool::Wind,
            KeyCode::BracketLeft => c.radius = (c.radius - 1).max(ui::MIN_RADIUS),
            KeyCode::BracketRight => c.radius = (c.radius + 1).min(ui::MAX_RADIUS),
            // Time: freeze the world, nudge it one tick, or run it slower or
            // faster.
            KeyCode::Space => c.paused = !c.paused,
            KeyCode::Period => self.input.step = true,
            KeyCode::Minus => c.slower(),
            KeyCode::Equal => c.faster(),
            KeyCode::KeyC => {
                if let Some(state) = &mut self.state {
                    state.sim.clear();
                }
            }
            // The world: build it again from the seed in the box, or from a
            // fresh one.
            KeyCode::KeyG => self.generate(),
            KeyCode::KeyR => self.randomize(),
            // Capture: a still, or a recording until pressed again.
            KeyCode::KeyS => self.screenshot(),
            KeyCode::KeyV => self.toggle_recording(),
            _ => {}
        }
    }

    /// Build the chosen world from the seed in the box and put it in place of
    /// whatever was there. A world script that fails says so on the panel,
    /// the way a plugin that fails to load does, and the old world stays.
    fn generate(&mut self) {
        let Some(state) = &mut self.state else {
            return;
        };
        let c = &self.input.controls;
        let (world, seed) = (c.world, c.seed_value());
        match self.plugins.generate(world, seed) {
            Ok(cells) => state.sim.load(&cells),
            Err(err) => {
                log::error!("world failed: {err}");
                self.status = err;
            }
        }
    }

    /// Roll a new seed into the box and build the chosen world from it.
    fn randomize(&mut self) {
        self.input.controls.set_seed(ui::random_seed());
        self.generate();
    }

    /// Load the user's plugins from [`PLUGIN_DIR`], and say on the panel how
    /// many there were, or what went wrong with the last one that failed.
    fn load_plugin_dir(&mut self) {
        let loaded = self.plugins.load_dir();
        if loaded.is_empty() {
            return;
        }
        self.status =
            match loaded.iter().rev().find_map(|(label, result)| {
                result.as_ref().err().map(|err| format!("{label}: {err}"))
            }) {
                Some(failure) => failure,
                None => format!(
                    "Loaded {} from {PLUGIN_DIR}/. {DROP_HINT}",
                    if loaded.len() == 1 {
                        "1 plugin".to_string()
                    } else {
                        format!("{} plugins", loaded.len())
                    }
                ),
            };
    }

    /// Run one script, put what it registered into the world, and say on the
    /// panel how that went.
    fn load_plugin(&mut self, path: &Path) {
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        match self.plugins.load_file(path) {
            Ok(report) => {
                log::info!("loaded plugin {}: {report}", path.display());
                self.status = format!("Loaded {label}: {report}.");
            }
            Err(err) => {
                log::error!("plugin {} failed: {err}", path.display());
                self.status = format!("{label}: {err}");
            }
        }
        // Whatever the script managed to register before any failure is in
        // the registry, so the tables are refreshed either way.
        if let Some(state) = &mut self.state {
            state.sim.set_tables(&self.plugins.registry());
        }
        self.apply_commands();
    }

    /// Do what the scripts have queued up. Nothing happens until the world
    /// exists; the queue simply waits.
    fn apply_commands(&mut self) {
        let Some(state) = &mut self.state else {
            return;
        };
        for command in self.plugins.take_commands() {
            command.apply(&mut state.sim);
        }
    }
}

/// Open the window and run until it closes, with `script`, a name and its
/// text, running in it if there is one.
pub fn run(script: Option<(String, String)>) {
    log::info!(
        "Controls: use the panel, or press 1-9 to pick a material in picker order \
         (1=Sand 2=Stone 3=Water 4=Lava 5=Soil, then the plugin materials)  0/Backspace=Erase  \
         W=wind tool (sweep to blow a gust)  [ ]=brush size  Space=pause  .=step one tick  \
         - ==slower/faster  \
         C=clear  G=build the world again  R=build it from a new seed  \
         S=screenshot  V=start/stop recording  \
         (hold left mouse to draw). \
         Drop a .js file on the window to load a plugin."
    );

    let event_loop = EventLoop::new().expect("build event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(script);
    event_loop.run_app(&mut app).expect("run event loop");
}
