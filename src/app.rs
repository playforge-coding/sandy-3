//! The window, the input, and the event loop: the glue between winit, the GPU
//! state, egui and the simulation.
//!
//! Window events go to egui first. Whatever it does not want, which is clicks
//! outside the panel and keys pressed with nothing focused, drives the brush and
//! the shortcuts. Each redraw runs egui, steps the world, and hands the
//! tessellated panel to [`State::render`] to layer over the scene.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::gpu::State;
use crate::materials::{EMPTY, LAVA, SAND, SOIL, STONE, WATER};
use crate::ui;

/// The window size the app opens at, in logical pixels. Twice as wide as it is
/// tall, matching the grid, so nothing is stretched out of shape at the start.
const WINDOW_SIZE: (f64, f64) = (1100.0, 620.0);

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

/// Where the mouse is and what it is doing, plus the [`ui::Controls`] the panel
/// and the shortcuts share.
struct Input {
    cursor: (f64, f64),
    drawing: bool,
    /// The cell the wind tool was over on the previous painted frame, so a drag
    /// yields a direction. `None` at the start of a stroke, so the first frame
    /// only notes where it began and blows nothing.
    last_wind: Option<(i32, i32)>,
    controls: ui::Controls,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            cursor: (0.0, 0.0),
            drawing: false,
            last_wind: None,
            controls: ui::Controls::default(),
        }
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
    input: Input,
    /// egui's context: fonts, memory and layout. Cheap to clone, since it is an
    /// `Arc` inside.
    egui_ctx: egui::Context,
    /// Per-window input translation for egui, built once the window exists.
    egui_state: Option<egui_winit::State>,
}

impl App {
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

        self.state = Some(pollster::block_on(State::new(window)));
        self.ensure_egui();
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
                    self.input.last_wind = None; // a fresh stroke has no direction yet
                } else {
                    self.input.drawing = false;
                }
            }

            // A touchscreen draws exactly as the mouse does. The position rides
            // on the event itself rather than arriving separately, so the cursor
            // is updated here too.
            WindowEvent::Touch(touch) => {
                self.input.cursor = (touch.location.x, touch.location.y);
                match touch.phase {
                    TouchPhase::Started => {
                        self.input.drawing = !consumed;
                        self.input.last_wind = None;
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

            WindowEvent::RedrawRequested => self.redraw(),

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

        if self.input.drawing {
            let brush = self.input.controls.brush;
            let tool = self.input.controls.tool;
            let material = self.input.controls.material;
            let state = self.state.as_mut().unwrap();
            let (gx, gy) = state.cursor_to_grid(self.input.cursor);
            match tool {
                ui::Tool::Paint => state.sim.paint_disk(gx, gy, brush, material),
                ui::Tool::Wind => {
                    // Blow a gust the way the cursor has swept since the last
                    // frame. The first frame of a stroke only notes where it is.
                    if let Some((px, py)) = self.input.last_wind {
                        let dvx = (gx - px) as f32 * WIND_DRAG_GAIN;
                        let dvy = (gy - py) as f32 * WIND_DRAG_GAIN;
                        let radius = (brush * GUST_SCALE).max(MIN_GUST_RADIUS);
                        state.sim.add_wind_disk(gx, gy, radius, dvx, dvy);
                    }
                    self.input.last_wind = Some((gx, gy));
                }
            }
        }

        // Run egui for this frame. The window handle is cloned so it does not
        // hold a borrow of `self.state` while that is mutated below.
        let window = self.state.as_ref().unwrap().window_arc();
        let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(&window);
        let ctx = self.egui_ctx.clone();
        let mut actions = ui::Actions::default();
        let full_output = ctx.run_ui(raw_input, |ui| {
            actions = ui::draw(ui.ctx(), &mut self.input.controls);
        });
        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output);
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        if let Some(state) = &mut self.state {
            if actions.clear {
                state.sim.clear();
            }
            state.update();
            state.render(
                paint_jobs,
                full_output.textures_delta,
                full_output.pixels_per_point,
            );
        }
    }

    fn handle_key(&mut self, code: KeyCode) {
        let c = &mut self.input.controls;
        match code {
            // Material selection. These match the order in `materials::table`.
            KeyCode::Digit1 => c.material = SAND,
            KeyCode::Digit2 => c.material = STONE,
            KeyCode::Digit3 => c.material = WATER,
            KeyCode::Digit4 => c.material = LAVA,
            KeyCode::Digit5 => c.material = SOIL,
            KeyCode::Digit0 | KeyCode::Backspace => c.material = EMPTY,
            // The wind tool: sweep the cursor to blow a gust.
            KeyCode::KeyW => c.tool = ui::Tool::Wind,
            KeyCode::BracketLeft => c.brush = (c.brush - 1).max(1),
            KeyCode::BracketRight => c.brush = (c.brush + 1).min(60),
            KeyCode::KeyC => {
                if let Some(state) = &mut self.state {
                    state.sim.clear();
                }
            }
            _ => {}
        }
        // Choosing a material means painting with it.
        if matches!(
            code,
            KeyCode::Digit1
                | KeyCode::Digit2
                | KeyCode::Digit3
                | KeyCode::Digit4
                | KeyCode::Digit5
                | KeyCode::Digit0
                | KeyCode::Backspace
        ) {
            c.tool = ui::Tool::Paint;
        }
    }
}

/// Open the window and run until it closes.
pub fn run() {
    env_logger::init();

    log::info!(
        "Controls: use the panel, or press 1=Sand 2=Stone 3=Water 4=Lava 5=Soil  0/Backspace=Erase  \
         W=wind tool (sweep to blow a gust)  [ ]=brush size  C=clear  (hold left mouse to draw)"
    );

    let event_loop = EventLoop::new().expect("build event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run event loop");
}
