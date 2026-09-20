//! Sandy 3 — a falling-sand world whose physics run on the GPU.
//!
//! - `materials` — one file per material, plus the tables the kernels read.
//! - `kernels`   — the simulation, as compute kernels written in Rust.
//! - `sim`       — the GPU buffers the world lives in, and the order of passes.
//! - `gpu`       — wgpu setup and the per-frame draw.
//! - `ui`        — the egui control panel.
//! - `app`       — the window, the input and the event loop.

mod app;
mod gpu;
mod kernels;
mod materials;
mod sim;
mod ui;

pub use app::run;
