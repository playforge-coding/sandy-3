//! Sandy 3 — a falling-sand world whose physics run on the GPU.
//!
//! - `materials` — one file per material, plus the tables the kernels read.
//! - `kernels`   — the simulation, as compute kernels written in Rust.
//! - `sim`       — the GPU buffers the world lives in, and the order of passes.
//! - `gpu`       — wgpu setup and the per-frame draw.
//! - `plugins`   — Lua scripts that add materials, tools and worlds.
//! - `worldgen`  — the canvas and the noise a world script builds with.
//! - `ui`        — the egui control panel.
//! - `app`       — the window, the input and the event loop.

mod app;
mod gpu;
mod kernels;
mod materials;
mod plugins;
mod sim;
mod ui;
mod worldgen;

pub use app::run;
