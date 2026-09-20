//! Sandy 3 — a falling-sand world whose physics run on the GPU.
//!
//! - `materials` — one file per material, plus the tables the kernels read.
//! - `kernels`   — the simulation, as compute kernels written in Rust.
//! - `sim`       — the GPU buffers the world lives in, and the order of passes.
//! - `gpu`       — wgpu setup and the per-frame draw.
//! - `ui`        — the egui control panel.
//! - `app`       — the window, the input and the event loop.
//!
//! The same code runs on the desktop and in a browser. The only difference is
//! how the GPU device is asked for: the desktop can block and wait for it, and
//! the browser cannot, so there it is requested off to one side and handed back
//! through the event loop (see [`app`]).

mod app;
mod gpu;
mod kernels;
mod materials;
mod sim;
mod ui;

pub use app::run;

// The browser entry point, called as soon as the wasm module has loaded.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    run();
}
