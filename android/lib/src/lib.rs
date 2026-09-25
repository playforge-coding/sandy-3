//! The Android entry point. `NativeActivity` loads this library, named in
//! the manifest under `android/app`, and calls `android_main` with the app it
//! has made; the game itself is in the main crate. Built with
//! `cargo ndk` (see the README), which is the only way this crate is ever
//! built: on any other platform it is empty.

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: sandy_3::AndroidApp) {
    sandy_3::run_android(app);
}
