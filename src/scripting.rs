//! The control API: a JavaScript script that drives the game, with a window
//! or without one.
//!
//! A plugin (see [`crate::plugins`]) adds things to the game and waits to be
//! used. A control script is the other way round: it *is* the user, for as
//! long as it runs. It paints, blows, steps the world, reads it back, takes
//! screenshots, records, and can load a plugin and drive its brushes the way
//! the mouse would. It is meant for tests, which is why every answer is
//! exact and every file is written before the call returns, and it suits
//! anything else that wants to run the game by remote: an automation, a
//! language model, or someone tinkering from a terminal.
//!
//! # How a script reaches the world
//!
//! Every function on the `sim` object is a plain call that does its work
//! before it returns: `sim.step(10)` has run ten ticks by the time the next
//! line runs, and `sim.count("Sand")` is a number. The script never holds
//! the simulation itself. Each call goes through [`Host`], which is whatever
//! owns the simulation this run, and an error there, because a material does
//! not exist or a file cannot be written, is thrown at the line that asked,
//! so it reads like any other error and `try`/`catch` catches it.
//!
//! The one call that cannot always be answered on the spot is `sim.frame`:
//! with a window, letting a frame go by means going back to the event loop.
//! So `sim.frame` is the one function that returns a promise, and a script
//! awaits it. [`Script::advance`] runs the script until it is waiting on a
//! frame, or done, and the app calls it again on the next frame. Headless
//! there is no event loop, so a frame is a sixtieth of a second on a clock of
//! the host's own (see [`Clock`]) and the promise is already settled when
//! the script gets it.
//!
//! # How a call finds the host
//!
//! The functions on `sim` are made once, when the script starts, and live in
//! the engine; the host is borrowed from the app for the length of one
//! [`Script::advance`]. The two meet through a [`Slot`] the functions share:
//! `advance` puts a pointer to the host in it before running the script and
//! takes it out after, and a `sim` function takes the pointer out for the
//! length of its call, so nothing else can reach the host while it does.
//! That is what keeps a brush's `onDrag`, which a `sim.stroke` runs, from
//! calling `sim` itself, and what makes `sim` refuse to work at all when the
//! script is not the one running.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rquickjs::class::{Trace, Tracer};
use rquickjs::function::Opt;
use rquickjs::promise::PromiseState;
use rquickjs::{
    Class, Context, Ctx, Function, JsLifetime, Module, Object, Persistent, Promise, Value,
};

use crate::app::wind_tool;
use crate::capture::{self, Capture};
use crate::gpu::{Renderer, TICK_DT};
use crate::materials::{MaterialId, Registry};
use crate::plugins::{
    self, Kind, Plugins, Sink, Stroke, clamp_radius, describe, fail, optional, resolve,
};
use crate::sim::{GRID_H, GRID_W, Simulation};
use crate::ui::{self, Controls, MAX_SPEED, MIN_SPEED, Tool};

/// Where a script comes from, as the command line says it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    File(PathBuf),
    /// Code given on the command line with `-e`.
    Inline(String),
    /// Standard input, given as `-`.
    Stdin,
}

impl Source {
    /// The script's text, and a name for it, which is what the engine puts
    /// in front of the line number in an error message.
    pub fn read(&self) -> Result<(String, String), String> {
        match self {
            Source::File(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|err| format!("could not read {}: {err}", path.display()))?;
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                Ok((name, text))
            }
            Source::Inline(code) => Ok(("-e".to_string(), code.clone())),
            Source::Stdin => {
                let mut text = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
                    .map_err(|err| format!("could not read the script from stdin: {err}"))?;
                Ok(("stdin".to_string(), text))
            }
        }
    }
}

/// A JavaScript value kept outside the engine's context between calls.
type Kept<T> = Persistent<T>;

/// A control script, running.
pub struct Script {
    // Everything that points into the engine comes before the context, so it
    // is dropped first.
    /// The script's name and text, until the first advance runs it.
    source: Option<(String, String)>,
    /// The promise the script's evaluation settles, once it has started: it
    /// resolves when the script runs off the end and rejects when it throws.
    done: Option<Kept<Promise<'static>>>,
    /// What the `sim` functions share with this.
    slot: Rc<Slot>,
    context: Context,
}

/// The meeting point of the `sim` functions and the host; see the module
/// documentation.
#[derive(Default)]
struct Slot {
    /// The host, while a [`Script::advance`] is running and no `sim` call
    /// has it.
    host: Cell<Option<NonNull<Host<'static>>>>,
    /// Whether `sim.quit()` has been called.
    quit: Cell<bool>,
    /// The `sim.frame` promises still waiting on frames.
    frames: RefCell<Vec<Waiting>>,
}

/// A `sim.frame` call waiting on a window.
struct Waiting {
    /// How many frames have still to go by.
    left: u32,
    /// What settles the promise the script is awaiting.
    resolve: Kept<Function<'static>>,
}

impl Slot {
    /// Count a frame gone by, and hand back the resolvers of the waits that
    /// are over.
    fn due(&self) -> Vec<Kept<Function<'static>>> {
        let mut ready = Vec::new();
        self.frames.borrow_mut().retain_mut(|waiting| {
            waiting.left = waiting.left.saturating_sub(1);
            if waiting.left == 0 {
                ready.push(waiting.resolve.clone());
                false
            } else {
                true
            }
        });
        ready
    }
}

/// How a script stands after an [`Script::advance`].
#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    /// It is waiting on a frame; call again on the next one.
    Running,
    /// It ran to the end.
    Finished,
    /// It asked to end, and for the window to close.
    Quit,
    /// It threw, and this is the message.
    Failed(String),
}

impl Drop for Script {
    fn drop(&mut self) {
        // The `sim` functions in the engine hold the slot too, so what the
        // slot holds of the engine's is let go of here rather than waiting
        // for them.
        self.slot.frames.borrow_mut().clear();
    }
}

impl Script {
    /// Put the `sim` object in `plugins`' engine, with `print` going to
    /// stdout, and compile `source` so a syntax error shows up here rather
    /// than on the first advance. Nothing runs until then.
    pub fn start(plugins: &Plugins, name: &str, source: &str) -> Result<Script, String> {
        let context = plugins.context().clone();
        let slot = Rc::new(Slot::default());
        context.with(|ctx| {
            (|| -> rquickjs::Result<()> {
                // A plugin's `console` goes to the log, where nobody is
                // watching. A control script is being run from a terminal or
                // a test, and its output is the point.
                plugins::install_globals(&ctx, Sink::Stdout)?;
                install_sim(&ctx, &slot)
            })()
            .map_err(|err| describe(&ctx, err))?;
            Module::declare(ctx.clone(), name, source)
                .map(drop)
                .map_err(|err| describe(&ctx, err))
        })?;
        Ok(Script {
            source: Some((name.to_string(), source.to_string())),
            done: None,
            slot,
            context,
        })
    }

    /// Run the script until it finishes, fails, or has to wait for a frame,
    /// with `host` answering its calls as it goes.
    pub fn advance(&mut self, host: &mut Host<'_>) -> Status {
        // The host's lifetime is erased so the slot can name its type. The
        // pointer is only ever followed inside the `with` below, which is
        // inside this call, for which `host` is borrowed exclusively and not
        // otherwise used; see `with_host` for the other half of the argument.
        let pointer = NonNull::from(host).cast::<Host<'static>>();
        self.slot.host.set(Some(pointer));
        let context = self.context.clone();
        let status = context.with(|ctx| self.turn(&ctx));
        self.slot.host.set(None);
        status
    }

    /// One advance, inside the engine.
    fn turn(&mut self, ctx: &Ctx<'_>) -> Status {
        if let Some((name, source)) = self.source.take() {
            match Module::declare(ctx.clone(), name, source).and_then(|module| module.eval()) {
                Ok((_, done)) => self.done = Some(Persistent::save(ctx, done)),
                Err(err) => return Status::Failed(describe(ctx, err)),
            }
        } else {
            for resolve in self.slot.due() {
                let settled = resolve
                    .restore(ctx)
                    .and_then(|resolve| resolve.call::<_, ()>(()));
                if let Err(err) = settled {
                    return Status::Failed(describe(ctx, err));
                }
            }
        }
        // A settled promise only wakes its awaiter when the engine runs its
        // jobs, and that is where the script runs on from an `await`.
        while ctx.execute_pending_job() {}

        if self.slot.quit.get() {
            return Status::Quit;
        }
        let done = match self.done.clone().map(|done| done.restore(ctx)) {
            Some(Ok(done)) => done,
            Some(Err(err)) => return Status::Failed(describe(ctx, err)),
            None => return Status::Failed("the script never started".to_string()),
        };
        match done.state() {
            PromiseState::Resolved => Status::Finished,
            PromiseState::Rejected => match done.result::<()>() {
                Some(Err(err)) => Status::Failed(describe(ctx, err)),
                _ => Status::Failed("the script failed".to_string()),
            },
            PromiseState::Pending if self.slot.frames.borrow().is_empty() => Status::Failed(
                "the script is waiting on something that never comes; sim.frame is the one \
                 thing that can be awaited"
                    .to_string(),
            ),
            PromiseState::Pending => Status::Running,
        }
    }
}

/// Run `f` with the host, for a `sim` function. The pointer is taken out of
/// the slot for the length of the call and put back after, so a nested call
/// (a brush's `onDrag` run by `sim.stroke`, say) finds nothing and is
/// refused, and so does any call made when no advance is running.
fn with_host<R>(
    slot: &Slot,
    ctx: &Ctx<'_>,
    f: impl FnOnce(&mut Host<'_>) -> rquickjs::Result<R>,
) -> rquickjs::Result<R> {
    let Some(pointer) = slot.host.take() else {
        return Err(fail(
            ctx,
            "sim is for the control script alone, while it runs; a brush, a tool or a world \
             cannot use it",
        ));
    };
    // SAFETY: the pointer was made in `Script::advance` from a `&mut Host`
    // that is borrowed for the whole of that call, and it is set only for
    // the length of that call, so the host is alive. The slot is emptied
    // while the reference exists, so this is the only reference to the host
    // until it is put back, and `advance` itself does not touch the host
    // while the script runs.
    let host = unsafe { &mut *pointer.as_ptr() };
    let result = f(host);
    slot.host.set(Some(pointer));
    result
}

/// Put the `sim` object in the globals, every function on it going through
/// `slot` to the host.
fn install_sim<'js>(ctx: &Ctx<'js>, slot: &Rc<Slot>) -> rquickjs::Result<()> {
    let sim = Object::new(ctx.clone())?;
    sim.set("width", GRID_W)?;
    sim.set("height", GRID_H)?;

    /// One `sim` function: a closure over the host, with the call's
    /// arguments after it.
    macro_rules! define {
        ($name:literal, |$host:ident, $c:ident $(, $arg:ident : $ty:ty)*| $body:expr) => {{
            let slot = slot.clone();
            sim.set(
                $name,
                Function::new(ctx.clone(), move |$c: Ctx<'js> $(, $arg: $ty)*| {
                    with_host(&slot, &$c, |$host| $body)
                })?,
            )?;
        }};
    }

    // Time.
    define!("step", |host, ctx, ticks: Opt<u32>| host
        .step(&ctx, ticks.0));
    {
        let slot = slot.clone();
        sim.set(
            "frame",
            Function::new(ctx.clone(), move |ctx: Ctx<'js>, frames: Opt<u32>| {
                let waits = slot.clone();
                with_host(&slot, &ctx, |host| host.frame(&ctx, &waits, frames.0))
            })?,
        )?;
    }
    define!("pause", |host, ctx| {
        host.controls.paused = true;
        Ok(())
    });
    define!("resume", |host, ctx| {
        host.controls.paused = false;
        Ok(())
    });
    define!("paused", |host, ctx| Ok(host.controls.paused));
    define!("speed", |host, ctx, multiplier: Opt<f64>| host
        .speed(&ctx, multiplier.0));
    define!("ticks", |host, ctx| Ok(host.sim.ticks()));

    // Changing the world.
    define!("paint", |host,
                      ctx,
                      x: f64,
                      y: f64,
                      radius: f64,
                      material: Value<'js>| {
        let material = host.material(&ctx, &material)?;
        host.sim
            .paint_disk(coord(x), coord(y), clamp_radius(radius), material);
        Ok(())
    });
    define!("fill", |host,
                     ctx,
                     x0: f64,
                     y0: f64,
                     x1: f64,
                     y1: f64,
                     material: Value<'js>| {
        let material = host.material(&ctx, &material)?;
        host.sim
            .fill(coord(x0), coord(y0), coord(x1), coord(y1), material);
        Ok(())
    });
    define!("wind", |host,
                     ctx,
                     x: f64,
                     y: f64,
                     radius: f64,
                     dvx: f64,
                     dvy: f64| {
        host.sim.add_wind_disk(
            coord(x),
            coord(y),
            clamp_radius(radius),
            dvx as f32,
            dvy as f32,
        );
        Ok(())
    });
    define!("clear", |host, ctx| {
        host.sim.clear();
        Ok(())
    });
    define!("generate", |host, ctx, world: String, seed: Opt<u32>| host
        .generate(&ctx, &world, seed.0));

    // Driving a brush or a tool the way the mouse does, and the panel.
    define!("stroke", |host, ctx, spec: Object<'js>| host
        .stroke(&ctx, &spec));
    define!("pick", |host, ctx, spec: Object<'js>| host
        .pick(&ctx, &spec));

    // Reading the world back.
    define!("snapshot", |host, ctx| {
        let snapshot = host.snapshot(&ctx)?;
        Class::instance(ctx.clone(), snapshot)
    });
    define!("get", |host, ctx, x: f64, y: f64| {
        let word = host
            .sim
            .read_cell(coord(x), coord(y))
            .map_err(|err| fail(&ctx, err))?;
        Ok(word.map(|word| (word & 0xff) as MaterialId))
    });
    define!("count", |host, ctx, material: Value<'js>| {
        let material = host.material(&ctx, &material)?;
        let cells = host.sim.read_cells().map_err(|err| fail(&ctx, err))?;
        Ok(cells
            .iter()
            .filter(|&&word| (word & 0xff) as MaterialId == material)
            .count())
    });

    // Files.
    define!("screenshot", |host, ctx, path: Opt<String>| host
        .screenshot(&ctx, path.0));
    define!("record", |host, ctx, path: Opt<String>| host
        .record(&ctx, path.0));
    define!("stop", |host, ctx| host.stop(&ctx));
    define!("plugin", |host, ctx, path: String| host.plugin(&ctx, &path));

    // What there is.
    define!("materials", |host, ctx| host.materials(&ctx));
    define!("brushes", |host, ctx| Ok(host.plugins.names(Kind::Brush)));
    define!("tools", |host, ctx| Ok(host.plugins.names(Kind::Tool)));
    define!("worlds", |host, ctx| Ok(host.plugins.world_names()));

    // The end. Nothing after the call runs, unless the script catches the
    // throw, and even then the flag is set and the script ends with this
    // advance.
    {
        let slot = slot.clone();
        sim.set(
            "quit",
            Function::new(ctx.clone(), move |ctx: Ctx<'js>| -> rquickjs::Result<()> {
                slot.quit.set(true);
                Err(fail(&ctx, "the script asked to quit"))
            })?,
        )?;
    }

    ctx.globals().set("sim", sim)
}

/// The clock a headless run keeps, since it has no frames to keep time by.
/// A tick or a frame is a sixtieth of a second of it, and a recording takes
/// its frames from it the way a recording with a window takes them from
/// real time.
pub struct Clock {
    /// The time it is, for the recording.
    now: Instant,
    /// The part of a tick a frame has banked and not yet spent, as
    /// [`crate::gpu::State::update`] banks real time.
    accumulator: f64,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    pub fn new() -> Self {
        Clock {
            now: Instant::now(),
            accumulator: 0.0,
        }
    }
}

/// Everything a script's calls are answered from. With a window it is
/// borrowed from the app for the length of one [`Script::advance`]; headless
/// it is [`crate::headless::Headless`]'s own parts.
pub struct Host<'a> {
    pub plugins: &'a mut Plugins,
    pub sim: &'a mut Simulation,
    pub renderer: &'a mut Renderer,
    pub capture: &'a mut Capture,
    pub controls: &'a mut Controls,
    /// The clock to keep, when there is no window keeping one. With it the
    /// host is headless: a frame is a step of this clock, and a recording is
    /// captured here rather than by the event loop.
    pub clock: Option<&'a mut Clock>,
}

/// How a brush or a tool is chosen for a stroke.
enum Which {
    Brush(usize),
    Tool(usize),
    Wind,
}

impl Host<'_> {
    /// `sim.step`: that many ticks, one by default, there and then.
    fn step(&mut self, ctx: &Ctx<'_>, ticks: Option<u32>) -> rquickjs::Result<()> {
        for _ in 0..ticks.unwrap_or(1) {
            self.tick(ctx)?;
        }
        Ok(())
    }

    /// `sim.frame`: let that many frames go by, one by default. Headless the
    /// frames are stepped here and the promise comes back settled; with a
    /// window the promise waits in `slot` for the event loop to draw them.
    fn frame<'js>(
        &mut self,
        ctx: &Ctx<'js>,
        slot: &Slot,
        frames: Option<u32>,
    ) -> rquickjs::Result<Promise<'js>> {
        let frames = frames.unwrap_or(1);
        let (promise, resolve, _reject) = ctx.promise()?;
        if self.clock.is_some() {
            for _ in 0..frames {
                self.headless_frame(ctx)?;
            }
            resolve.call::<_, ()>(())?;
        } else if frames == 0 {
            resolve.call::<_, ()>(())?;
        } else {
            slot.frames.borrow_mut().push(Waiting {
                left: frames,
                resolve: Persistent::save(ctx, resolve),
            });
        }
        Ok(promise)
    }

    /// `sim.speed`: the panel's speed, set first if a multiplier was given.
    fn speed(&mut self, ctx: &Ctx<'_>, multiplier: Option<f64>) -> rquickjs::Result<f64> {
        if let Some(multiplier) = multiplier {
            if !(multiplier.is_finite() && multiplier > 0.0) {
                return Err(fail(ctx, "speed should be a positive number"));
            }
            self.controls.speed = multiplier.clamp(MIN_SPEED, MAX_SPEED);
        }
        Ok(self.controls.speed)
    }

    /// `sim.generate`: build a world by name, from the seed given or a
    /// rolled one, and say which seed it was. The panel follows, so Generate
    /// afterwards builds the same world again.
    fn generate(&mut self, ctx: &Ctx<'_>, world: &str, seed: Option<u32>) -> rquickjs::Result<u32> {
        let index = self
            .plugins
            .world_index(world)
            .ok_or_else(|| fail(ctx, format!("there is no world called '{world}'")))?;
        let seed = seed.unwrap_or_else(ui::random_seed);
        let cells = self
            .plugins
            .generate_in(ctx, index, seed)
            .map_err(|err| fail(ctx, err))?;
        self.sim.load(&cells);
        self.controls.world = index;
        self.controls.set_seed(seed);
        Ok(seed)
    }

    /// `sim.screenshot`: save a picture, to `path` or to a fresh file in the
    /// captures folder, written before this returns.
    fn screenshot(&mut self, ctx: &Ctx<'_>, path: Option<String>) -> rquickjs::Result<String> {
        let path = match path {
            Some(path) => PathBuf::from(path),
            None => self
                .capture
                .fresh_file(self.controls.screenshot_format.extension())
                .map_err(|err| fail(ctx, format!("could not make the captures folder: {err}")))?,
        };
        let frame = self.renderer.frame_now().map_err(|err| fail(ctx, err))?;
        capture::save_still(&path, &frame)
            .map_err(|err| fail(ctx, format!("could not save {}: {err}", path.display())))?;
        log::info!("saved {}", path.display());
        Ok(path.to_string_lossy().into_owned())
    }

    /// `sim.record`: start a recording, to `path` or to a fresh file.
    fn record(&mut self, ctx: &Ctx<'_>, path: Option<String>) -> rquickjs::Result<String> {
        let started = self.now();
        let path = self
            .capture
            .start_recording(
                path.map(PathBuf::from),
                self.controls.recording_format,
                started,
            )
            .map_err(|err| fail(ctx, err))?;
        log::info!("recording to {}", path.display());
        Ok(path.to_string_lossy().into_owned())
    }

    /// `sim.stop`: end the recording. Headless, every frame is already in,
    /// so the file is only waiting to be finished, and the script wants it
    /// whole. With a window the last frames are still on their way back
    /// from the GPU and the event loop delivers them.
    fn stop(&mut self, ctx: &Ctx<'_>) -> rquickjs::Result<String> {
        let path = self
            .capture
            .stop_recording()
            .ok_or_else(|| fail(ctx, "no recording is running"))?;
        if self.clock.is_some() {
            match self.capture.wait_report() {
                Ok(line) => log::info!("{line}"),
                Err(line) => return Err(fail(ctx, line)),
            }
        }
        Ok(path.to_string_lossy().into_owned())
    }

    /// `sim.plugin`: load a plugin file, as dropping it on the window would,
    /// and say what it registered.
    fn plugin(&mut self, ctx: &Ctx<'_>, path: &str) -> rquickjs::Result<String> {
        let report = self
            .plugins
            .load_file_in(ctx, Path::new(path))
            .map_err(|err| fail(ctx, err))?;
        self.sim.set_tables(&self.plugins.registry());
        self.apply_commands();
        Ok(report.to_string())
    }

    /// One tick, and headless a sixtieth of a second of the clock with it.
    fn tick(&mut self, ctx: &Ctx<'_>) -> rquickjs::Result<()> {
        self.sim.step();
        if self.clock.is_some() {
            self.elapse(ctx, TICK_DT)?;
        }
        Ok(())
    }

    /// One frame with no window: the world runs at the panel's speed, as
    /// [`crate::gpu::State::update`] runs it, and the clock moves on a frame.
    fn headless_frame(&mut self, ctx: &Ctx<'_>) -> rquickjs::Result<()> {
        let rate = self.controls.rate();
        let clock = self.clock.as_mut().expect("a frame is stepped headless");
        clock.accumulator += rate * TICK_DT;
        // A whisker under, so a quarter speed ticks on the fourth frame and
        // not the fifth because of a rounding.
        while clock.accumulator >= TICK_DT - 1e-9 {
            clock.accumulator -= TICK_DT;
            self.sim.step();
        }
        self.elapse(ctx, TICK_DT)
    }

    /// Move the headless clock on, and if the recording wants a frame at the
    /// new time, draw the world and hand it over.
    fn elapse(&mut self, ctx: &Ctx<'_>, seconds: f64) -> rquickjs::Result<()> {
        let Some(clock) = self.clock.as_mut() else {
            return Ok(());
        };
        clock.now += Duration::from_secs_f64(seconds);
        if let Some(wanted) = self.capture.plan(clock.now) {
            let frame = self.renderer.frame_now().map_err(|err| fail(ctx, err))?;
            self.capture.taken(wanted);
            self.capture.deliver(frame);
        }
        Ok(())
    }

    /// The time it is, for a recording: the clock's, or real time.
    fn now(&self) -> Instant {
        self.clock
            .as_ref()
            .map(|clock| clock.now)
            .unwrap_or_else(Instant::now)
    }

    fn material(&self, ctx: &Ctx<'_>, value: &Value<'_>) -> rquickjs::Result<MaterialId> {
        resolve(ctx, &self.plugins.registry(), value, "material")
    }

    /// Do what the plugins' brushes and tools have queued.
    fn apply_commands(&mut self) {
        for command in self.plugins.take_commands() {
            command.apply(self.sim);
        }
    }

    /// `sim.stroke`: drive a brush or a tool along a path, a frame per point,
    /// as the mouse would. The spec names a `brush` or a `tool` (`Wind` is
    /// the game's own), or neither for what the panel has picked, and may
    /// give the `material` and `radius`, which also default to the panel's.
    fn stroke<'js>(&mut self, ctx: &Ctx<'js>, spec: &Object<'js>) -> rquickjs::Result<()> {
        let brush: Option<String> = optional(ctx, spec, "brush", None)?;
        let tool: Option<String> = optional(ctx, spec, "tool", None)?;
        let material: Value = spec.get("material")?;
        let material = if material.type_of().is_void() {
            self.controls.material
        } else {
            self.material(ctx, &material)?
        };
        let radius: Option<f64> = optional(ctx, spec, "radius", None)?;
        let radius = radius.map(clamp_radius).unwrap_or(self.controls.radius);
        let path: Option<Vec<Vec<f64>>> = optional(ctx, spec, "path", None)?;
        let path = path.ok_or_else(|| fail(ctx, "'path' is required: a list of [x, y] points"))?;
        let points: Vec<(i32, i32)> = path
            .iter()
            .map(|point| match point[..] {
                [x, y] => Ok((coord(x), coord(y))),
                _ => Err(fail(ctx, "each point of the path is [x, y]")),
            })
            .collect::<rquickjs::Result<_>>()?;

        let which = match (brush, tool) {
            (Some(_), Some(_)) => {
                return Err(fail(ctx, "a stroke is with a brush or a tool, not both"));
            }
            (Some(name), None) => Which::Brush(self.index_of(ctx, Kind::Brush, &name)?),
            (None, Some(name)) if name.eq_ignore_ascii_case("wind") => Which::Wind,
            (None, Some(name)) => Which::Tool(self.index_of(ctx, Kind::Tool, &name)?),
            (None, None) => match self.controls.tool {
                Tool::Paint => Which::Brush(self.controls.brush),
                Tool::Wind => Which::Wind,
                Tool::Plugin(index) => Which::Tool(index),
            },
        };

        let mut last: Option<(i32, i32)> = None;
        for (x, y) in points {
            match which {
                Which::Wind => {
                    // The first frame of a sweep only notes where it began.
                    if let Some(from) = last {
                        wind_tool(self.sim, from, (x, y), radius);
                    }
                }
                Which::Brush(index) | Which::Tool(index) => {
                    let kind = match which {
                        Which::Brush(_) => Kind::Brush,
                        _ => Kind::Tool,
                    };
                    let (px, py) = last.unwrap_or((x, y));
                    let stroke = Stroke {
                        x,
                        y,
                        px,
                        py,
                        first: last.is_none(),
                        radius,
                        material,
                    };
                    self.plugins
                        .run_in(ctx, kind, index, stroke)
                        .map_err(|err| fail(ctx, err))?;
                    self.apply_commands();
                }
            }
            last = Some((x, y));
        }
        Ok(())
    }

    /// `sim.pick`: set what the panel has picked, as clicking it would.
    fn pick<'js>(&mut self, ctx: &Ctx<'js>, spec: &Object<'js>) -> rquickjs::Result<()> {
        let material: Value = spec.get("material")?;
        if !material.type_of().is_void() {
            self.controls.material = self.material(ctx, &material)?;
            self.controls.tool = Tool::Paint;
        }
        let brush: Option<String> = optional(ctx, spec, "brush", None)?;
        if let Some(name) = brush {
            self.controls.brush = self.index_of(ctx, Kind::Brush, &name)?;
            self.controls.tool = Tool::Paint;
        }
        let tool: Option<String> = optional(ctx, spec, "tool", None)?;
        if let Some(name) = tool {
            self.controls.tool = if name.eq_ignore_ascii_case("wind") {
                Tool::Wind
            } else {
                Tool::Plugin(self.index_of(ctx, Kind::Tool, &name)?)
            };
        }
        let radius: Option<f64> = optional(ctx, spec, "radius", None)?;
        if let Some(radius) = radius {
            self.controls.radius = clamp_radius(radius).clamp(ui::MIN_RADIUS, ui::MAX_RADIUS);
        }
        Ok(())
    }

    fn index_of(&self, ctx: &Ctx<'_>, kind: Kind, name: &str) -> rquickjs::Result<usize> {
        self.plugins.index_of(kind, name).ok_or_else(|| {
            fail(
                ctx,
                format!(
                    "there is no {} called '{name}'",
                    match kind {
                        Kind::Brush => "brush",
                        Kind::Tool => "tool",
                    }
                ),
            )
        })
    }

    /// The world and the wind as they are now, read back.
    fn snapshot(&self, ctx: &Ctx<'_>) -> rquickjs::Result<Snapshot> {
        let cells = self.sim.read_cells().map_err(|err| fail(ctx, err))?;
        let wind = self.sim.read_wind().map_err(|err| fail(ctx, err))?;
        Ok(Snapshot {
            width: self.sim.width,
            height: self.sim.height,
            ticks: self.sim.ticks(),
            cells: cells
                .iter()
                .map(|&word| (word & 0xff) as MaterialId)
                .collect(),
            wind,
            registry: self.plugins.registry().clone(),
        })
    }

    /// `sim.materials`: every material as a record, in id order.
    fn materials<'js>(&self, ctx: &Ctx<'js>) -> rquickjs::Result<Vec<Object<'js>>> {
        let registry = self.plugins.registry();
        registry
            .materials()
            .iter()
            .enumerate()
            .map(|(id, info)| {
                let entry = Object::new(ctx.clone())?;
                entry.set("id", id as MaterialId)?;
                entry.set("name", info.name)?;
                entry.set("color", info.color.to_vec())?;
                entry.set("jitter", info.jitter)?;
                entry.set("density", info.density)?;
                entry.set("mobile", info.mobile)?;
                entry.set("passable", info.passable)?;
                entry.set("liquid", info.liquid)?;
                entry.set("spread", info.spread)?;
                entry.set("windborne", info.windborne)?;
                entry.set("glow", info.glow)?;
                entry.set("draft", info.draft)?;
                Ok(entry)
            })
            .collect()
    }
}

/// A cell coordinate a script gave, which may be a float from its
/// arithmetic. Rounded down, as the world canvas rounds, and kept well
/// inside `i32` so a radius added to it cannot overflow.
fn coord(v: f64) -> i32 {
    const LIMIT: f64 = i32::MAX as f64 / 4.0;
    if v.is_finite() {
        v.floor().clamp(-LIMIT, LIMIT) as i32
    } else {
        // Off the grid, so anything painted here is clipped away.
        -LIMIT as i32
    }
}

/// The world at one moment, as `sim.snapshot()` hands it to a script: the
/// material of every cell and the wind over it, with the questions a test
/// asks of them. Coordinates are cells from the top left, and a material is
/// a name in any case or an id.
#[derive(JsLifetime)]
#[rquickjs::class]
pub struct Snapshot {
    width: u32,
    height: u32,
    ticks: u32,
    cells: Vec<MaterialId>,
    wind: Vec<[f32; 2]>,
    /// The materials as they were, so a name can still be looked up.
    registry: Registry,
}

/// Nothing in a snapshot is a JavaScript value, so there is nothing for the
/// garbage collector to follow.
impl<'js> Trace<'js> for Snapshot {
    fn trace<'a>(&self, _tracer: Tracer<'a, 'js>) {}
}

impl Snapshot {
    fn index(&self, x: f64, y: f64) -> Option<usize> {
        let (x, y) = (coord(x), coord(y));
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            None
        } else {
            Some((y as u32 * self.width + x as u32) as usize)
        }
    }

    fn material(&self, ctx: &Ctx<'_>, value: &Value<'_>) -> rquickjs::Result<MaterialId> {
        resolve(ctx, &self.registry, value, "material")
    }

    /// Every cell holding `material`, as `(x, y)`.
    fn places(&self, material: MaterialId) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.cells
            .iter()
            .enumerate()
            .filter(move |&(_, &m)| m == material)
            .map(|(i, _)| (i as u32 % self.width, i as u32 / self.width))
    }
}

#[rquickjs::methods]
impl Snapshot {
    #[qjs(get)]
    fn width(&self) -> u32 {
        self.width
    }

    #[qjs(get)]
    fn height(&self) -> u32 {
        self.height
    }

    #[qjs(get)]
    fn ticks(&self) -> u32 {
        self.ticks
    }

    /// The material at a cell, or undefined off the grid.
    fn get(&self, x: f64, y: f64) -> Option<MaterialId> {
        self.index(x, y).map(|i| self.cells[i])
    }

    /// How many cells hold a material.
    fn count(&self, ctx: Ctx<'_>, material: Value<'_>) -> rquickjs::Result<usize> {
        let material = self.material(&ctx, &material)?;
        Ok(self.cells.iter().filter(|&&m| m == material).count())
    }

    /// The smallest rectangle holding every cell of a material, as
    /// `[x0, y0, x1, y1]` with both corners included, or undefined.
    fn bounds(&self, ctx: Ctx<'_>, material: Value<'_>) -> rquickjs::Result<Option<Vec<u32>>> {
        let material = self.material(&ctx, &material)?;
        let mut bounds: Option<(u32, u32, u32, u32)> = None;
        for (x, y) in self.places(material) {
            bounds = Some(match bounds {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
        Ok(bounds.map(|(x0, y0, x1, y1)| vec![x0, y0, x1, y1]))
    }

    /// The topmost row holding a material, which is the smallest y, or
    /// undefined.
    fn highest(&self, ctx: Ctx<'_>, material: Value<'_>) -> rquickjs::Result<Option<u32>> {
        let material = self.material(&ctx, &material)?;
        Ok(self.places(material).map(|(_, y)| y).min())
    }

    /// The bottommost row holding a material, or undefined.
    fn lowest(&self, ctx: Ctx<'_>, material: Value<'_>) -> rquickjs::Result<Option<u32>> {
        let material = self.material(&ctx, &material)?;
        Ok(self.places(material).map(|(_, y)| y).max())
    }

    /// Where the middle of all of a material sits, as `[x, y]`, or undefined.
    fn center(&self, ctx: Ctx<'_>, material: Value<'_>) -> rquickjs::Result<Option<Vec<f64>>> {
        let material = self.material(&ctx, &material)?;
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
        for (x, y) in self.places(material) {
            sx += f64::from(x);
            sy += f64::from(y);
            n += 1;
        }
        Ok((n > 0).then(|| vec![sx / n as f64, sy / n as f64]))
    }

    /// The wind at a cell, as `[vx, vy]` in cells per tick, or undefined off
    /// the grid.
    fn wind(&self, x: f64, y: f64) -> Option<Vec<f64>> {
        self.index(x, y)
            .map(|i| vec![f64::from(self.wind[i][0]), f64::from(self.wind[i][1])])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless::Headless;

    /// Run `source` headless and say how it ended.
    fn run(source: &str) -> Result<(), String> {
        Headless::new().unwrap().run("test.js", source)
    }

    #[test]
    fn a_script_paints_steps_and_reads_the_world_back() {
        run(r#"
            assert(sim.width === 1000 && sim.height === 500);
            assert(sim.ticks() === 0);
            sim.fill(0, sim.height - 4, sim.width - 1, sim.height - 1, "Stone");
            sim.paint(500, 60, 14, "sand");
            const painted = sim.count("Sand");
            assert(painted > 500, "the brush put sand down: " + painted);
            assert(sim.get(500, 60) === sandy.find("Sand"));
            assert(sim.get(-1, 60) === undefined, "off the grid is undefined");

            sim.step(400);
            assert(sim.ticks() === 400);
            const world = sim.snapshot();
            assert(world.ticks === 400 && world.width === sim.width);
            assert(world.count("Sand") === painted, "nothing was lost on the way down");
            assert(world.highest("Sand") > 400, "the sand fell to the floor");
            assert(world.lowest("Sand") < sim.height - 4, "and stopped on it");
            const [x0, y0, x1, y1] = world.bounds("Sand");
            assert(x0 < 500 && x1 > 500 && y1 === sim.height - 5, `${x0} ${x1} ${y1}`);
            const [cx, cy] = world.center("Sand");
            assert(Math.abs(cx - 500) < 30 && cy > 400, `${cx} ${cy}`);
            assert(world.get(500, sim.height - 1) === sandy.find("Stone"));
            assert(world.get(sim.width, 0) === undefined);
            assert(world.highest("Lava") === undefined && world.count("lava") === 0);
            assert(world.bounds("Lava") === undefined, "no bounds for nothing");
            const [vx, vy] = world.wind(500, 250);
            assert(typeof vx === "number" && typeof vy === "number");
            assert(world.wind(-5, 0) === undefined);

            sim.clear();
            assert(sim.count("Sand") === 0 && sim.count("Stone") === 0);
        "#)
        .unwrap();
    }

    #[test]
    fn an_error_in_a_call_points_at_the_line_that_made_it() {
        let err = run("const x = 1;\nsim.paint(1, 2, 3, 'Unobtainium')").unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");
        assert!(err.contains("test.js:2"), "{err}");

        let err = run("sim.generate('Nowhere')").unwrap_err();
        assert!(err.contains("Nowhere"), "{err}");

        let err = run("sim.step('lots')").unwrap_err();
        assert!(err.contains("test.js:1"), "{err}");

        let err = run("throw new Error('on purpose')").unwrap_err();
        assert!(
            err.contains("on purpose") && err.contains("test.js:1"),
            "{err}"
        );

        let err = run("sim.paint(").unwrap_err();
        assert!(err.contains("test.js"), "a syntax error says where: {err}");

        // A refused call is an ordinary exception, so try/catch catches it.
        run(r#"
            let caught;
            try { sim.stop(); } catch (err) { caught = err; }
            assert(caught && caught.message.includes("no recording"), caught);
            assert(sim.count("Sand") === 0, "the script carried on");
        "#)
        .unwrap();

        // Awaiting anything but a frame never comes back.
        let err = run("await new Promise(() => {})").unwrap_err();
        assert!(err.contains("never comes"), "{err}");

        // A brush a stroke runs cannot drive the game itself.
        let err = run(r#"
            sandy.brush({ name: "Sneaky", onDrag: (t) => sim.step() });
            sim.stroke({ brush: "Sneaky", path: [[1, 1]] });
        "#)
        .unwrap_err();
        assert!(err.contains("control script"), "{err}");
        assert!(err.contains("test.js:3"), "{err}");
    }

    #[test]
    fn frames_run_the_clock_headless_at_the_panels_speed() {
        run(r#"
            await sim.frame(10);
            assert(sim.ticks() === 10, "a frame is a tick at full speed");
            sim.pause();
            assert(sim.paused());
            await sim.frame(10);
            assert(sim.ticks() === 10, "a paused world stands still");
            sim.resume();
            assert(!sim.paused());
            assert(sim.speed(2) === 2);
            await sim.frame(10);
            assert(sim.ticks() === 30, "double speed is two ticks a frame");
            assert(sim.speed(0.25) === 0.25);
            await sim.frame(4);
            assert(sim.ticks() === 31, "quarter speed is a tick every four frames");
            assert(sim.speed(100) === 4, "clamped to the panel's range");
            assert(sim.speed() === 4);
            await sim.frame();
            assert(sim.ticks() === 35);
            sim.step();
            assert(sim.ticks() === 36, "a step is one tick whatever the speed");
        "#)
        .unwrap();
        let err = run("sim.speed(0)").unwrap_err();
        assert!(err.contains("positive"), "{err}");
    }

    #[test]
    fn generate_builds_a_world_from_a_seed_and_the_panel_follows() {
        let mut headless = Headless::new().unwrap();
        headless
            .run(
                "world.js",
                r#"
                const names = sim.worlds();
                assert(names[0] === "Forest" && names.length === 5, names.join(","));
                assert(sim.generate("forest", 1337) === 1337);
                assert(sim.count("Stone") > 50000);
                assert(sim.count("Wood") > 100, "trees");
                const seed = sim.generate("Desert");
                assert(seed >= 0);
                assert(sim.count("Water") === 0, "the desert is dry");
                assert(sim.count("Sand") > 5000);
                "#,
            )
            .unwrap();
        let (world, seed) = headless.picked_world();
        assert_eq!(world, 3, "the panel is on the desert");
        assert!(!seed.is_empty());
    }

    #[test]
    fn a_stroke_drives_a_brush_and_a_tool_as_the_mouse_would() {
        run(r#"
            const brushes = sim.brushes();
            assert(brushes[0] === "Disk" && brushes[1] === "Spray", brushes.join(","));
            assert(sim.tools()[0] === "Fan");

            // Three frames of the spray, so at least three grains land.
            sim.stroke({ brush: "Spray", material: "Water", radius: 6,
                         path: [[100, 100], [104, 100], [108, 100]] });
            const world = sim.snapshot();
            const n = world.count("Water");
            assert(n >= 3, n);
            const [x0, y0, x1, y1] = world.bounds("Water");
            assert(x0 >= 93 && x1 <= 115 && y0 >= 93 && y1 <= 107, "inside the brush");

            // The panel's own pick is the default.
            sim.pick({ material: "Stone", brush: "Disk", radius: 4 });
            sim.stroke({ path: [[300, 300]] });
            assert(sim.count("Stone") > 40);

            // A tool: the fan blows up, so the air above the cursor rises.
            sim.stroke({ tool: "Fan", radius: 10, path: [[500, 400], [500, 400], [500, 400]] });
            const [, vy] = sim.snapshot().wind(500, 390);
            assert(vy < 0, "the fan is an updraft: " + vy);

            // The wind tool needs a sweep; a single point blows nothing.
            sim.clear();
            sim.stroke({ tool: "Wind", path: [[200, 250]] });
            let [vx] = sim.snapshot().wind(200, 250);
            assert(vx === 0, vx);
            sim.stroke({ tool: "Wind", path: [[200, 250], [210, 250]] });
            [vx] = sim.snapshot().wind(210, 250);
            assert(vx > 1, "a sweep to the right blows to the right: " + vx);

            // Picking a tool on the panel, then a stroke with no spec.
            sim.pick({ tool: "wind" });
            sim.pick({ tool: "Fan" });
        "#)
        .unwrap();

        for bad in [
            "sim.stroke({ brush: 'Nope', path: [[1, 1]] })",
            "sim.stroke({ tool: 'Nope', path: [[1, 1]] })",
            "sim.stroke({ brush: 'Disk', tool: 'Fan', path: [[1, 1]] })",
            "sim.stroke({ brush: 'Disk' })",
            "sim.stroke({ brush: 'Disk', path: [[1]] })",
            "sim.pick({ brush: 'Nope' })",
        ] {
            let err = run(bad).unwrap_err();
            assert!(err.contains("test.js:1"), "{bad}: {err}");
        }
    }

    #[test]
    fn a_screenshot_is_written_before_the_call_returns() {
        let dir = std::env::temp_dir().join(format!("sandy-script-shot-{}", std::process::id()));
        let path = dir.join("shot.png");
        run(&format!(
            r#"
            sim.fill(0, 400, 999, 499, "Lava");
            const where = sim.screenshot({path:?});
            assert(where === {path:?}, where);
            "#
        ))
        .unwrap();
        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let mut reader = ::png::Decoder::new(file).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (1000, 500));
        // The bottom of the picture is lava coloured: much more red than blue.
        let px = |x: usize, y: usize| {
            let i = (y * 1000 + x) * info.color_type.samples();
            (buf[i], buf[i + 1], buf[i + 2])
        };
        let (r, _, b) = px(500, 450);
        assert!(r > 150 && r > b + 50, "lava is red, not {:?}", px(500, 450));
        let (r, _, b) = px(500, 100);
        assert!(b > r + 50, "the sky is blue, not {:?}", px(500, 100));
        std::fs::remove_dir_all(&dir).unwrap();

        let err = run("sim.screenshot('shot.bmp')").unwrap_err();
        assert!(err.contains(".png"), "{err}");
    }

    #[test]
    fn a_headless_recording_takes_frames_off_the_clock() {
        let dir = std::env::temp_dir().join(format!("sandy-script-rec-{}", std::process::id()));
        let path = dir.join("clip.gif");
        run(&r#"
            sim.paint(500, 100, 20, "Sand");
            assert(sim.record(PATH) === PATH);
            let caught;
            try { sim.record(); } catch (err) { caught = err; }
            assert(caught && caught.message.includes("already"), caught);
            await sim.frame(20);
            sim.step(20);
            assert(sim.stop() === PATH);
            "#
        .replace("PATH", &format!("{path:?}")))
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"GIF89a"));
        // Forty sixtieths of a second at thirty frames a second is twenty
        // frames, each an image descriptor in the file.
        let descriptors = bytes.iter().filter(|&&b| b == b',').count();
        assert!(descriptors >= 15, "only {descriptors} image descriptors");
        std::fs::remove_dir_all(&dir).unwrap();

        // A recording left running is finished when the script ends.
        let path = dir.join("left.webp");
        run(&format!("sim.record({path:?}); await sim.frame(4)")).unwrap();
        assert!(std::fs::read(&path).unwrap().starts_with(b"RIFF"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_script_can_load_a_plugin_and_use_what_it_added() {
        let dir = std::env::temp_dir().join(format!("sandy-script-plugin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plugin = dir.join("ash.js");
        std::fs::write(
            &plugin,
            r#"
            sandy.material({ name: "Ash", color: [90, 90, 90], density: 60, mobile: true });
            sandy.brush({ name: "Dot", onDrag: (t) => sandy.paint(t.x, t.y, 0, t.material) });
            "#,
        )
        .unwrap();
        run(&r#"
            const report = sim.plugin(PLUGIN);
            assert(report.includes("1 material") && report.includes("1 brush"), report);
            const found = sim.materials().find((m) => m.name === "Ash");
            assert(found && found.density === 60 && found.mobile && found.color[0] === 90);
            assert(found.id === sandy.find("Ash"));
            sim.stroke({ brush: "Dot", material: "Ash", path: [[10, 10], [11, 10]] });
            assert(sim.count("Ash") === 2);
            // The registered material falls, so the tables reached the GPU.
            sim.step(30);
            assert(sim.snapshot().highest("Ash") > 20);
            // A control script can register things itself, too.
            sandy.material({ name: "Dust", color: [1, 1, 1], density: 30, mobile: true });
            "#
        .replace("PLUGIN", &format!("{plugin:?}")))
        .unwrap();
        let err = run(&format!("sim.plugin({:?})", dir.join("missing.js"))).unwrap_err();
        assert!(err.contains("could not read"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn with_a_window_a_frame_waits_for_the_next_advance_and_quit_ends_it() {
        // The same host without a clock is what the app builds: `frame`
        // cannot be answered until the event loop has drawn one.
        let mut headless = Headless::new().unwrap();
        let mut script = Script::start(
            &headless.plugins,
            "windowed.js",
            r#"
            sim.paint(10, 10, 0, "Sand");
            await sim.frame(2);
            sim.paint(20, 10, 0, "Sand");
            await sim.frame();
            sim.quit();
            throw new Error("never reached");
            "#,
        )
        .unwrap();
        let mut statuses = Vec::new();
        for _ in 0..10 {
            let status = script.advance(&mut headless.host(false));
            let done = status != Status::Running;
            statuses.push(status);
            if done {
                break;
            }
        }
        assert_eq!(
            statuses,
            [
                Status::Running,
                Status::Running,
                Status::Running,
                Status::Quit
            ]
        );
        assert_eq!(headless.sim.read_cells().unwrap()[10 * 1000 + 20] & 0xff, 1);

        // Once the script is gone, `sim` is still there but does nothing.
        drop(script);
        let err = headless.plugins.load("late.js", "sim.step()").unwrap_err();
        assert!(err.contains("control script"), "{err}");
    }
}
