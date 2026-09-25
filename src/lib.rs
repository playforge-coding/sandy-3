//! Sandy 3 — a falling-sand world whose physics run on the GPU.
//!
//! - `materials` — one file per material, plus the tables the kernels read.
//! - `kernels`   — the simulation, as compute kernels written in Rust.
//! - `sim`       — the GPU buffers the world lives in, and the order of passes.
//! - `gpu`       — wgpu setup and the per-frame draw.
//! - `capture`   — screenshots and recordings, and the files they are written to.
//! - `plugins`   — JavaScript plugins that add materials, tools and worlds.
//! - `worldgen`  — the canvas and the noise a world script builds with.
//! - `scripting` — the control API: a JavaScript script that drives the game.
//! - `headless`  — running a control script with no window.
//! - `cli`       — the command line.
//! - `ui`        — the egui control panel.
//! - `app`       — the window, the input and the event loop.
//! - `mobile`    — what is different on a phone: the folders, the log, and
//!   the Android entry point.

use std::process::ExitCode;

mod app;
mod capture;
mod cli;
mod gpu;
mod headless;
mod kernels;
mod materials;
mod mobile;
mod plugins;
mod scripting;
mod sim;
mod ui;
mod worldgen;

/// The Android side hands the app in through `android_main`; see
/// [`run_android`].
#[cfg(target_os = "android")]
pub use winit::platform::android::activity::AndroidApp;

/// Run the game on Android, on the activity the system has made. This is
/// what the `android/lib` crate's `android_main` calls, and it is the whole
/// of the Android entry point; the command line and the headless mode do not
/// apply there.
#[cfg(target_os = "android")]
pub fn run_android(app: AndroidApp) {
    mobile::run_android(app);
}

/// Read the command line and do as it says: open the window, with a script
/// running in it or not, or run a script headless. The status is what the
/// process exits with.
pub fn run() -> ExitCode {
    #[cfg(target_os = "ios")]
    mobile::setup_ios();
    #[cfg(not(target_os = "ios"))]
    env_logger::init();
    match cli::parse(std::env::args().skip(1)) {
        Ok(cli::Command::Help) => {
            println!("{}", cli::USAGE);
            ExitCode::SUCCESS
        }
        Ok(cli::Command::Headless { script }) => match headless::run(&script) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Ok(cli::Command::Window { script }) => {
            // Read the script before the window opens, so a missing file
            // is a line on the terminal rather than a line on the panel.
            let script = match script.as_ref().map(scripting::Source::read).transpose() {
                Ok(script) => script,
                Err(err) => {
                    eprintln!("{err}");
                    return ExitCode::FAILURE;
                }
            };
            app::run(script);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}\n\n{}", cli::USAGE);
            ExitCode::FAILURE
        }
    }
}
