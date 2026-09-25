//! What is different on a phone.
//!
//! The game is the same program on Android and iOS as on a desktop: the same
//! kernels, the same plugins, the same panel. What changes is around the
//! edges, and it is gathered here so the rest of the code can ask one
//! question, [`IS_MOBILE`], or call one function, rather than sprout
//! `cfg` branches of its own.
//!
//! - The world is smaller, and shaped to the screen: see
//!   [`crate::sim::Grid::for_screen`].
//! - The panel is laid out for a finger: see [`layout`].
//! - There is no working directory to speak of, so the process moves into
//!   the app's own folder before anything is read or written. The `plugins`
//!   and `captures` folders then live there, which on iOS is the app's
//!   Documents folder, visible in the Files app, and on Android the app's
//!   folder under `Android/data`, reachable over USB or with a file manager.
//! - The log goes to logcat on Android, where `adb logcat -s sandy` shows it.
//!   On iOS it goes to standard error, which the simulator shows when the
//!   app is launched with `--console`.
//!
//! Android has an entry point of its own, [`run_android`]: the app is a
//! shared library that the system's `NativeActivity` loads, and it calls
//! `android_main` rather than `main`. That function is in the small crate
//! under `android/lib`, and comes straight here.

use crate::ui;

/// Whether this is the phone build. Most of the code never asks; the world
/// size, the panel and the folders do.
pub const IS_MOBILE: bool = cfg!(any(target_os = "android", target_os = "ios"));

/// How far below the top of the screen the panel opens, in points. Neither
/// platform's windowing tells winit where the status bar or the notch ends,
/// so this is a figure that clears them on the phones there are: an iPhone's
/// Dynamic Island reaches down about sixty points, and an Android status bar
/// is a little under fifty on most. The panel can be dragged from there
/// anyway.
const TOP_INSET: f32 = if cfg!(target_os = "ios") { 60.0 } else { 48.0 };

/// How the panel should be laid out on this platform.
pub fn layout() -> ui::Layout {
    if IS_MOBILE {
        ui::Layout {
            touch: true,
            top_inset: TOP_INSET,
        }
    } else {
        ui::Layout::DESKTOP
    }
}

/// What the foot of the panel says about plugins when it has nothing more
/// recent to say. There is nothing to drop a file on with a phone; the
/// folder is the way in.
pub const DROP_HINT: &str = if IS_MOBILE {
    "Put a .js plugin in the app's plugins folder and open the game again."
} else {
    "Drop a .js file on the window to load a plugin."
};

/// Make `dir` the working directory, so the relative folders the game uses
/// land somewhere the app may write. A failure is logged and left at that:
/// the game still runs, and the panel says so if a capture cannot be saved.
#[cfg(any(target_os = "android", target_os = "ios"))]
fn move_into(dir: &std::path::Path) {
    match std::fs::create_dir_all(dir).and_then(|()| std::env::set_current_dir(dir)) {
        Ok(()) => log::info!("working in {}", dir.display()),
        Err(err) => log::error!("could not move into {}: {err}", dir.display()),
    }
}

/// The iOS side of what [`run_android`] does for Android: the log, and the
/// app's Documents folder as the working directory, which is the one the
/// Files app shows. Called from the ordinary `main` before anything else.
#[cfg(target_os = "ios")]
pub fn setup_ios() {
    // The simulator shows standard error, so the log is worth having on by
    // default here, where there is no shell to set `RUST_LOG` from.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // iOS points `HOME` at the app's own container.
    match std::env::var_os("HOME") {
        Some(home) => move_into(&std::path::PathBuf::from(home).join("Documents")),
        None => log::error!("no HOME to find the Documents folder from"),
    }
}

/// The Android entry point, reached from `android_main` in the `android/lib`
/// crate. Sets the log up to go to logcat, moves into the app's folder, and
/// opens the window on the activity the system has made.
#[cfg(target_os = "android")]
pub fn run_android(app: winit::platform::android::activity::AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("sandy"),
    );
    // The folder under `Android/data` if the device has one, which a file
    // manager can reach, and the private one otherwise.
    match app
        .external_data_path()
        .or_else(|| app.internal_data_path())
    {
        Some(dir) => move_into(&dir),
        None => log::error!("the system gave the app no folder to work in"),
    }
    crate::app::run_android(app);
}
