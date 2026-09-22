//! Running a control script with no window.
//!
//! `sandy-3 --headless script.lua` brings the GPU up without a surface, the
//! way the tests do, loads the plugins, and runs the script (see
//! [`crate::scripting`]) against an empty world until it finishes. There is
//! no event loop and nothing to draw, so the script gets every answer at
//! once and time is a clock of its own that moves when the script steps the
//! world. The process exits when the script does, with a status of one if
//! the script failed, which is what makes it a test runner.

use crate::capture::Capture;
use crate::gpu::Renderer;
use crate::plugins::Plugins;
use crate::scripting::{Clock, Host, Script, Source, Status};
use crate::sim::Simulation;
use crate::ui::Controls;

/// The GPU, the world and the plugins, with no window over them.
pub struct Headless {
    pub plugins: Plugins,
    pub sim: Simulation,
    pub renderer: Renderer,
    pub capture: Capture,
    pub controls: Controls,
    clock: Clock,
}

impl Headless {
    /// Bring the GPU up and build an empty world that knows the built-in
    /// plugins and any in the plugins folder.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|err| {
            format!(
                "no GPU to run on: {err}. This needs a GPU with compute shaders, so Vulkan, Metal \
                 or D3D12"
            )
        })?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sandy headless"),
            ..Default::default()
        }))
        .map_err(|err| format!("could not open the GPU: {err}"))?;

        let mut plugins = Plugins::new();
        plugins.load_builtin();
        plugins.load_dir();
        let sim = Simulation::new(&device, &queue, &plugins.registry());
        let renderer = Renderer::new(&device, &queue, &sim);
        Ok(Headless {
            plugins,
            sim,
            renderer,
            capture: Capture::default(),
            controls: Controls::default(),
            clock: Clock::new(),
        })
    }

    /// Run `source` to the end, named `name` in its error messages. A
    /// recording the script left running is finished before this returns,
    /// whether the script ended well or not.
    pub fn run(&mut self, name: &str, source: &str) -> Result<(), String> {
        let result = self.drive(name, source);
        if self.capture.is_recording() {
            self.capture.stop_recording();
            match self.capture.wait_report() {
                Ok(line) => log::info!("{line}"),
                Err(line) => log::error!("{line}"),
            }
        }
        result
    }

    fn drive(&mut self, name: &str, source: &str) -> Result<(), String> {
        let mut script = Script::start(&self.plugins, name, source)?;
        loop {
            match script.advance(&mut self.host(true)) {
                // Headless, nothing is ever left waiting, but there is no
                // harm in asking again.
                Status::Running => {}
                Status::Finished | Status::Quit => return Ok(()),
                Status::Failed(err) => return Err(err),
            }
        }
    }

    /// The parts, borrowed as a script's host: with the clock, which is the
    /// headless way, or without it, which is how the app does it and what
    /// the tests use to try that path.
    pub fn host(&mut self, with_clock: bool) -> Host<'_> {
        Host {
            plugins: &mut self.plugins,
            sim: &mut self.sim,
            renderer: &mut self.renderer,
            capture: &mut self.capture,
            controls: &mut self.controls,
            clock: with_clock.then_some(&mut self.clock),
        }
    }

    /// The world the panel would be showing as picked, and the seed in its
    /// box, for the tests.
    #[cfg(test)]
    pub fn picked_world(&self) -> (usize, String) {
        (self.controls.world, self.controls.seed.clone())
    }
}

/// The command line's headless mode: read the script, run it, and say what
/// went wrong if anything did.
pub fn run(source: &Source) -> Result<(), String> {
    let (name, text) = source.read()?;
    let mut headless = Headless::new()?;
    headless.run(&name, &text)
}
