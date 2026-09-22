//! The control API: a Lua script that drives the game, with a window or
//! without one.
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
//! It never does, directly. The script runs as a Lua coroutine, and every
//! function on its `sim` table (the Lua half, in `scripting.lua`) yields a
//! request: the name of the operation and its arguments. [`Script::advance`]
//! catches the yield, hands the request to a [`Host`], which is whatever
//! owns the simulation this run, and resumes the coroutine with the answer.
//! To the script it reads as a plain call that returns a value, and the Lua
//! state and the GPU state never have to know about each other, which is the
//! same rule the plugins keep.
//!
//! The one request the host cannot always answer on the spot is `frame`:
//! with a window, letting a frame go by means going back to the event loop.
//! So [`Script::advance`] can also come back with the script still waiting,
//! and the app calls it again on the next frame. Headless there is no event
//! loop, so a frame is a sixtieth of a second on a clock of the host's own
//! (see [`Clock`]) and the script never has to wait for anything.
//!
//! Errors travel the other way: a request the host refuses, because a
//! material does not exist or a file cannot be written, is resumed into the
//! script as a Lua error raised at the line that asked, so it reads like
//! any other Lua error and `pcall` catches it if the script wants.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mlua::{
    FromLuaMulti, IntoLuaMulti, Lua, MultiValue, Table, Thread, UserData, UserDataFields,
    UserDataMethods, Value, Variadic,
};

use crate::app::wind_tool;
use crate::capture::{self, Capture};
use crate::gpu::{Renderer, TICK_DT};
use crate::materials::{MaterialId, Registry};
use crate::plugins::{
    Kind, Plugins, Stroke, clamp_radius, describe, optional, plain, resolve, runtime,
};
use crate::sim::Simulation;
use crate::ui::{self, Controls, MAX_SPEED, MIN_SPEED, Tool};

/// The Lua half of the API: the `sim` table, which turns each call into a
/// request for the host.
pub const PRELUDE: &str = include_str!("scripting.lua");

/// What Lua calls the prelude in a traceback. An error a request raises
/// passes through it, and the line worth telling the user is the script's,
/// so [`crate::plugins::describe`] skips this name.
pub(crate) const PRELUDE_NAME: &str = "sim";

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
    /// The script's text, and a name for it, which is what Lua puts in
    /// front of the line number in an error message.
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

/// A control script, running.
pub struct Script {
    thread: Thread,
    /// A request the host could not answer at once, tried again on the next
    /// [`Script::advance`].
    pending: Option<Pending>,
}

/// What a script is waiting on.
#[derive(Clone, Copy, Debug)]
enum Pending {
    /// This many frames have still to go by.
    Frames(u32),
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
    /// It raised an error, which is the message.
    Failed(String),
}

/// The host's answer to a request.
enum Reply {
    /// The results, now.
    Now(MultiValue),
    /// Not yet; ask again next frame.
    Later(Pending),
    /// The script is done.
    Halt,
}

/// Whether a pending request has been satisfied.
enum Poll {
    Ready,
    Wait(Pending),
}

impl Script {
    /// Compile `source` as a coroutine in `plugins`' interpreter, with the
    /// `sim` table in scope and `print` going to stdout, and leave it ready
    /// to run. A syntax error shows up here.
    pub fn start(plugins: &Plugins, name: &str, source: &str) -> Result<Script, String> {
        let lua = plugins.lua();
        (|| -> mlua::Result<()> {
            let sim: Table = lua
                .load(PRELUDE)
                .set_name(format!("={PRELUDE_NAME}"))
                .eval()?;
            lua.globals().set("sim", sim)?;
            // A plugin's `print` goes to the log, where nobody is watching. A
            // control script is being run from a terminal or a test, and its
            // output is the point, so it goes to stdout, flushed line by line.
            lua.globals().set(
                "print",
                lua.create_function(|_, args: Variadic<Value>| {
                    let parts: mlua::Result<Vec<String>> =
                        args.iter().map(|v| v.to_string()).collect();
                    let mut out = std::io::stdout().lock();
                    let _ = writeln!(out, "{}", parts?.join("\t"));
                    let _ = out.flush();
                    Ok(())
                })?,
            )?;
            Ok(())
        })()
        .map_err(|err| describe(&err))?;
        let thread = plugins.thread(name, source)?;
        Ok(Script {
            thread,
            pending: None,
        })
    }

    /// Run the script until it finishes, fails, or has to wait for a frame,
    /// answering its requests from `host` as it goes.
    pub fn advance(&mut self, host: &mut Host) -> Status {
        let lua = host.plugins.lua().clone();
        let mut reply = match self.pending.take() {
            Some(pending) => match host.poll(pending) {
                Poll::Wait(pending) => {
                    self.pending = Some(pending);
                    return Status::Running;
                }
                Poll::Ready => answer(&lua, Ok(MultiValue::new())),
            },
            None => MultiValue::new(),
        };
        loop {
            let yielded = match self.thread.resume::<MultiValue>(reply) {
                Ok(values) => values,
                Err(err) => return Status::Failed(describe(&err)),
            };
            if !self.thread.is_resumable() {
                return Status::Finished;
            }
            let mut values = yielded.into_vec().into_iter();
            let op = match values.next() {
                Some(Value::String(op)) => op.to_string_lossy(),
                _ => {
                    return Status::Failed(
                        "the script yielded on its own; only the sim functions can".to_string(),
                    );
                }
            };
            let args = MultiValue::from_vec(values.collect());
            match host.handle(&lua, &op, args) {
                Ok(Reply::Now(values)) => reply = answer(&lua, Ok(values)),
                Ok(Reply::Later(pending)) => {
                    self.pending = Some(pending);
                    return Status::Running;
                }
                Ok(Reply::Halt) => return Status::Quit,
                Err(err) => reply = answer(&lua, Err(plain(&err))),
            }
        }
    }
}

/// What the script is resumed with: `true` and the results, or `false` and
/// the message, which the Lua side raises as an error.
fn answer(lua: &Lua, result: Result<MultiValue, String>) -> MultiValue {
    let mut values = match result {
        Ok(values) => {
            let mut all = vec![Value::Boolean(true)];
            all.extend(values.into_vec());
            all
        }
        Err(message) => vec![
            Value::Boolean(false),
            lua.create_string(message)
                .map(Value::String)
                .unwrap_or(Value::Nil),
        ],
    };
    values.shrink_to_fit();
    MultiValue::from_vec(values)
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

/// Everything a script's requests are answered from. With a window it is
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
    /// Whether a request that had to wait is satisfied now.
    fn poll(&mut self, pending: Pending) -> Poll {
        match pending {
            Pending::Frames(left) if left > 1 => Poll::Wait(Pending::Frames(left - 1)),
            Pending::Frames(_) => Poll::Ready,
        }
    }

    /// Answer one request. An error here is resumed into the script as a
    /// Lua error.
    fn handle(&mut self, lua: &Lua, op: &str, args: MultiValue) -> mlua::Result<Reply> {
        match op {
            "step" => {
                let ticks: Option<u32> = take(lua, args)?;
                for _ in 0..ticks.unwrap_or(1) {
                    self.tick()?;
                }
                now(lua, ())
            }
            "frame" => {
                let frames: Option<u32> = take(lua, args)?;
                let frames = frames.unwrap_or(1);
                if self.clock.is_some() {
                    for _ in 0..frames {
                        self.headless_frame()?;
                    }
                    now(lua, ())
                } else if frames == 0 {
                    now(lua, ())
                } else {
                    Ok(Reply::Later(Pending::Frames(frames)))
                }
            }
            "pause" => {
                self.controls.paused = true;
                now(lua, ())
            }
            "resume" => {
                self.controls.paused = false;
                now(lua, ())
            }
            "paused" => now(lua, self.controls.paused),
            "speed" => {
                let multiplier: Option<f64> = take(lua, args)?;
                if let Some(multiplier) = multiplier {
                    if !(multiplier.is_finite() && multiplier > 0.0) {
                        return Err(runtime("speed should be a positive number"));
                    }
                    self.controls.speed = multiplier.clamp(MIN_SPEED, MAX_SPEED);
                }
                now(lua, self.controls.speed)
            }
            "ticks" => now(lua, self.sim.ticks()),

            "paint" => {
                let (x, y, radius, material): (f64, f64, f64, Value) = take(lua, args)?;
                let material = self.material(&material)?;
                self.sim
                    .paint_disk(coord(x), coord(y), clamp_radius(radius), material);
                now(lua, ())
            }
            "fill" => {
                let (x0, y0, x1, y1, material): (f64, f64, f64, f64, Value) = take(lua, args)?;
                let material = self.material(&material)?;
                self.sim
                    .fill(coord(x0), coord(y0), coord(x1), coord(y1), material);
                now(lua, ())
            }
            "wind" => {
                let (x, y, radius, dvx, dvy): (f64, f64, f64, f64, f64) = take(lua, args)?;
                self.sim.add_wind_disk(
                    coord(x),
                    coord(y),
                    clamp_radius(radius),
                    dvx as f32,
                    dvy as f32,
                );
                now(lua, ())
            }
            "clear" => {
                self.sim.clear();
                now(lua, ())
            }
            "generate" => {
                let (world, seed): (String, Option<u32>) = take(lua, args)?;
                let index = self
                    .plugins
                    .world_index(&world)
                    .ok_or_else(|| runtime(format!("there is no world called '{world}'")))?;
                let seed = seed.unwrap_or_else(ui::random_seed);
                let cells = self.plugins.generate(index, seed).map_err(runtime)?;
                self.sim.load(&cells);
                // The panel follows, so Generate afterwards builds the same
                // world again.
                self.controls.world = index;
                self.controls.set_seed(seed);
                now(lua, seed)
            }
            "stroke" => {
                let spec: Table = take(lua, args)?;
                self.stroke(&spec)?;
                now(lua, ())
            }
            "pick" => {
                let spec: Table = take(lua, args)?;
                self.pick(&spec)?;
                now(lua, ())
            }

            "snapshot" => {
                let snapshot = self.snapshot()?;
                now(lua, snapshot)
            }
            "get" => {
                let (x, y): (f64, f64) = take(lua, args)?;
                let word = self.sim.read_cell(coord(x), coord(y)).map_err(runtime)?;
                now(lua, word.map(|word| (word & 0xff) as MaterialId))
            }
            "count" => {
                let material: Value = take(lua, args)?;
                let material = self.material(&material)?;
                let cells = self.sim.read_cells().map_err(runtime)?;
                let count = cells
                    .iter()
                    .filter(|&&word| (word & 0xff) as MaterialId == material)
                    .count();
                now(lua, count)
            }

            "screenshot" => {
                let path: Option<String> = take(lua, args)?;
                let path = match path {
                    Some(path) => PathBuf::from(path),
                    None => self
                        .capture
                        .fresh_file(self.controls.screenshot_format.extension())
                        .map_err(|err| {
                            runtime(format!("could not make the captures folder: {err}"))
                        })?,
                };
                let frame = self.renderer.frame_now().map_err(runtime)?;
                capture::save_still(&path, &frame)
                    .map_err(|err| runtime(format!("could not save {}: {err}", path.display())))?;
                log::info!("saved {}", path.display());
                now(lua, path.to_string_lossy().into_owned())
            }
            "record" => {
                let path: Option<String> = take(lua, args)?;
                let started = self.now();
                let path = self
                    .capture
                    .start_recording(
                        path.map(PathBuf::from),
                        self.controls.recording_format,
                        started,
                    )
                    .map_err(runtime)?;
                log::info!("recording to {}", path.display());
                now(lua, path.to_string_lossy().into_owned())
            }
            "stop" => {
                let path = self
                    .capture
                    .stop_recording()
                    .ok_or_else(|| runtime("no recording is running"))?;
                // Headless, every frame is already in, so the file is only
                // waiting to be finished, and the script wants it whole.
                // With a window the last frames are still on their way back
                // from the GPU and the event loop delivers them.
                if self.clock.is_some() {
                    match self.capture.wait_report() {
                        Ok(line) => log::info!("{line}"),
                        Err(line) => return Err(runtime(line)),
                    }
                }
                now(lua, path.to_string_lossy().into_owned())
            }
            "plugin" => {
                let path: String = take(lua, args)?;
                let report = self.plugins.load_file(Path::new(&path)).map_err(runtime)?;
                self.sim.set_tables(&self.plugins.registry());
                self.apply_commands();
                now(lua, report.to_string())
            }

            "materials" => {
                let list = self.materials(lua)?;
                now(lua, list)
            }
            "brushes" => now(lua, self.plugins.names(Kind::Brush)),
            "tools" => now(lua, self.plugins.names(Kind::Tool)),
            "worlds" => now(lua, self.plugins.world_names()),

            "quit" => Ok(Reply::Halt),
            other => Err(runtime(format!("sim has nothing called '{other}'"))),
        }
    }

    /// One tick, and headless a sixtieth of a second of the clock with it.
    fn tick(&mut self) -> mlua::Result<()> {
        self.sim.step();
        if self.clock.is_some() {
            self.elapse(TICK_DT)?;
        }
        Ok(())
    }

    /// One frame with no window: the world runs at the panel's speed, as
    /// [`crate::gpu::State::update`] runs it, and the clock moves on a frame.
    fn headless_frame(&mut self) -> mlua::Result<()> {
        let rate = self.controls.rate();
        let clock = self.clock.as_mut().expect("a frame is stepped headless");
        clock.accumulator += rate * TICK_DT;
        // A whisker under, so a quarter speed ticks on the fourth frame and
        // not the fifth because of a rounding.
        while clock.accumulator >= TICK_DT - 1e-9 {
            clock.accumulator -= TICK_DT;
            self.sim.step();
        }
        self.elapse(TICK_DT)
    }

    /// Move the headless clock on, and if the recording wants a frame at the
    /// new time, draw the world and hand it over.
    fn elapse(&mut self, seconds: f64) -> mlua::Result<()> {
        let Some(clock) = self.clock.as_mut() else {
            return Ok(());
        };
        clock.now += Duration::from_secs_f64(seconds);
        if let Some(wanted) = self.capture.plan(clock.now) {
            let frame = self.renderer.frame_now().map_err(runtime)?;
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

    fn material(&self, value: &Value) -> mlua::Result<MaterialId> {
        resolve(&self.plugins.registry(), value, "material")
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
    fn stroke(&mut self, spec: &Table) -> mlua::Result<()> {
        let brush: Option<String> = optional(spec, "brush", None)?;
        let tool: Option<String> = optional(spec, "tool", None)?;
        let material: Value = spec.get("material")?;
        let material = if material.is_nil() {
            self.controls.material
        } else {
            self.material(&material)?
        };
        let radius: Option<f64> = optional(spec, "radius", None)?;
        let radius = radius.map(clamp_radius).unwrap_or(self.controls.radius);
        let path: Option<Vec<Vec<f64>>> = optional(spec, "path", None)?;
        let path = path.ok_or_else(|| runtime("'path' is required: a list of {x, y} points"))?;
        let points: Vec<(i32, i32)> = path
            .iter()
            .map(|point| match point[..] {
                [x, y] => Ok((coord(x), coord(y))),
                _ => Err(runtime("each point of the path is {x, y}")),
            })
            .collect::<mlua::Result<_>>()?;

        let which = match (brush, tool) {
            (Some(_), Some(_)) => {
                return Err(runtime("a stroke is with a brush or a tool, not both"));
            }
            (Some(name), None) => Which::Brush(self.index_of(Kind::Brush, &name)?),
            (None, Some(name)) if name.eq_ignore_ascii_case("wind") => Which::Wind,
            (None, Some(name)) => Which::Tool(self.index_of(Kind::Tool, &name)?),
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
                    self.plugins.run(kind, index, stroke).map_err(runtime)?;
                    self.apply_commands();
                }
            }
            last = Some((x, y));
        }
        Ok(())
    }

    /// `sim.pick`: set what the panel has picked, as clicking it would.
    fn pick(&mut self, spec: &Table) -> mlua::Result<()> {
        let material: Value = spec.get("material")?;
        if !material.is_nil() {
            self.controls.material = self.material(&material)?;
            self.controls.tool = Tool::Paint;
        }
        let brush: Option<String> = optional(spec, "brush", None)?;
        if let Some(name) = brush {
            self.controls.brush = self.index_of(Kind::Brush, &name)?;
            self.controls.tool = Tool::Paint;
        }
        let tool: Option<String> = optional(spec, "tool", None)?;
        if let Some(name) = tool {
            self.controls.tool = if name.eq_ignore_ascii_case("wind") {
                Tool::Wind
            } else {
                Tool::Plugin(self.index_of(Kind::Tool, &name)?)
            };
        }
        let radius: Option<f64> = optional(spec, "radius", None)?;
        if let Some(radius) = radius {
            self.controls.radius = clamp_radius(radius).clamp(ui::MIN_RADIUS, ui::MAX_RADIUS);
        }
        Ok(())
    }

    fn index_of(&self, kind: Kind, name: &str) -> mlua::Result<usize> {
        self.plugins.index_of(kind, name).ok_or_else(|| {
            runtime(format!(
                "there is no {} called '{name}'",
                match kind {
                    Kind::Brush => "brush",
                    Kind::Tool => "tool",
                }
            ))
        })
    }

    /// The world and the wind as they are now, read back.
    fn snapshot(&self) -> mlua::Result<Snapshot> {
        let cells = self.sim.read_cells().map_err(runtime)?;
        let wind = self.sim.read_wind().map_err(runtime)?;
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
    fn materials(&self, lua: &Lua) -> mlua::Result<Table> {
        let registry = self.plugins.registry();
        let list = lua.create_table_with_capacity(registry.materials().len(), 0)?;
        for (id, info) in registry.materials().iter().enumerate() {
            let entry = lua.create_table()?;
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
            list.raw_set(id + 1, entry)?;
        }
        Ok(list)
    }
}

/// The arguments of a request, as the types the operation takes.
fn take<T: FromLuaMulti>(lua: &Lua, args: MultiValue) -> mlua::Result<T> {
    T::from_lua_multi(args, lua)
}

/// An answer for the script, now.
fn now(lua: &Lua, values: impl IntoLuaMulti) -> mlua::Result<Reply> {
    Ok(Reply::Now(values.into_lua_multi(lua)?))
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
pub struct Snapshot {
    width: u32,
    height: u32,
    ticks: u32,
    cells: Vec<MaterialId>,
    wind: Vec<[f32; 2]>,
    /// The materials as they were, so a name can still be looked up.
    registry: Registry,
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

    fn material(&self, value: &Value) -> mlua::Result<MaterialId> {
        resolve(&self.registry, value, "material")
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

impl UserData for Snapshot {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("width", |_, s| Ok(s.width));
        fields.add_field_method_get("height", |_, s| Ok(s.height));
        fields.add_field_method_get("ticks", |_, s| Ok(s.ticks));
    }

    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // The material at a cell, or nil off the grid.
        methods.add_method("get", |_, s, (x, y): (f64, f64)| {
            Ok(s.index(x, y).map(|i| s.cells[i]))
        });
        // How many cells hold a material.
        methods.add_method("count", |_, s, material: Value| {
            let material = s.material(&material)?;
            Ok(s.cells.iter().filter(|&&m| m == material).count())
        });
        // The smallest rectangle holding every cell of a material, as
        // `x0, y0, x1, y1` with both corners included, or nothing.
        methods.add_method("bounds", |_, s, material: Value| {
            let material = s.material(&material)?;
            let mut bounds: Option<(u32, u32, u32, u32)> = None;
            for (x, y) in s.places(material) {
                bounds = Some(match bounds {
                    None => (x, y, x, y),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                });
            }
            Ok(match bounds {
                Some((x0, y0, x1, y1)) => Variadic::from_iter([x0, y0, x1, y1]),
                None => Variadic::new(),
            })
        });
        // The topmost row holding a material, which is the smallest y, or nil.
        methods.add_method("highest", |_, s, material: Value| {
            let material = s.material(&material)?;
            Ok(s.places(material).map(|(_, y)| y).min())
        });
        // The bottommost row holding a material, or nil.
        methods.add_method("lowest", |_, s, material: Value| {
            let material = s.material(&material)?;
            Ok(s.places(material).map(|(_, y)| y).max())
        });
        // Where the middle of all of a material sits, as `x, y`, or nothing.
        methods.add_method("center", |_, s, material: Value| {
            let material = s.material(&material)?;
            let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
            for (x, y) in s.places(material) {
                sx += x as f64;
                sy += y as f64;
                n += 1;
            }
            Ok(if n == 0 {
                Variadic::new()
            } else {
                Variadic::from_iter([sx / n as f64, sy / n as f64])
            })
        });
        // The wind at a cell, as `vx, vy` in cells per tick, or nothing off
        // the grid.
        methods.add_method("wind", |_, s, (x, y): (f64, f64)| {
            Ok(match s.index(x, y) {
                Some(i) => Variadic::from_iter([s.wind[i][0] as f64, s.wind[i][1] as f64]),
                None => Variadic::new(),
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless::Headless;

    /// Run `source` headless and say how it ended.
    fn run(source: &str) -> Result<(), String> {
        Headless::new().unwrap().run("test.lua", source)
    }

    #[test]
    fn a_script_paints_steps_and_reads_the_world_back() {
        run(r#"
            assert(sim.width == 1000 and sim.height == 500)
            assert(sim.ticks() == 0)
            sim.fill(0, sim.height - 4, sim.width - 1, sim.height - 1, "Stone")
            sim.paint(500, 60, 14, "sand")
            local painted = sim.count("Sand")
            assert(painted > 500, "the brush put sand down: " .. painted)
            assert(sim.get(500, 60) == sandy.find("Sand"))
            assert(sim.get(-1, 60) == nil, "off the grid is nil")

            sim.step(400)
            assert(sim.ticks() == 400)
            local world = sim.snapshot()
            assert(world.ticks == 400 and world.width == sim.width)
            assert(world:count("Sand") == painted, "nothing was lost on the way down")
            assert(world:highest("Sand") > 400, "the sand fell to the floor")
            assert(world:lowest("Sand") < sim.height - 4, "and stopped on it")
            local x0, y0, x1, y1 = world:bounds("Sand")
            assert(x0 < 500 and x1 > 500 and y1 == sim.height - 5, x0 .. " " .. x1 .. " " .. y1)
            local cx, cy = world:center("Sand")
            assert(math.abs(cx - 500) < 30 and cy > 400, cx .. " " .. cy)
            assert(world:get(500, sim.height - 1) == sandy.find("Stone"))
            assert(world:get(sim.width, 0) == nil)
            assert(world:highest("Lava") == nil and world:count("lava") == 0)
            assert(select('#', world:bounds("Lava")) == 0, "no bounds for nothing")
            local vx, vy = world:wind(500, 250)
            assert(type(vx) == "number" and type(vy) == "number")
            assert(select('#', world:wind(-5, 0)) == 0)

            sim.clear()
            assert(sim.count("Sand") == 0 and sim.count("Stone") == 0)
        "#)
        .unwrap();
    }

    #[test]
    fn an_error_in_a_request_points_at_the_line_that_asked() {
        let err = run("local x = 1\nsim.paint(1, 2, 3, 'Unobtainium')").unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");
        assert!(err.contains("test.lua:2"), "{err}");

        let err = run("sim.generate('Nowhere')").unwrap_err();
        assert!(err.contains("Nowhere"), "{err}");

        let err = run("sim.step('lots')").unwrap_err();
        assert!(err.contains("test.lua:1"), "{err}");

        let err = run("error('on purpose')").unwrap_err();
        assert!(
            err.contains("on purpose") && err.contains("test.lua:1"),
            "{err}"
        );

        let err = run("sim.paint(").unwrap_err();
        assert!(err.contains("test.lua"), "a syntax error says where: {err}");

        // A refused request is an ordinary Lua error, so pcall catches it.
        run(r#"
            local ok, err = pcall(sim.stop)
            assert(not ok and err:find("no recording"), err)
            assert(sim.count("Sand") == 0, "the script carried on")
        "#)
        .unwrap();

        // Yielding on one's own is not a request.
        let err = run("coroutine.yield(1)").unwrap_err();
        assert!(err.contains("yielded on its own"), "{err}");
    }

    #[test]
    fn frames_run_the_clock_headless_at_the_panels_speed() {
        run(r#"
            sim.frame(10)
            assert(sim.ticks() == 10, "a frame is a tick at full speed")
            sim.pause()
            assert(sim.paused())
            sim.frame(10)
            assert(sim.ticks() == 10, "a paused world stands still")
            sim.resume()
            assert(not sim.paused())
            assert(sim.speed(2) == 2)
            sim.frame(10)
            assert(sim.ticks() == 30, "double speed is two ticks a frame")
            assert(sim.speed(0.25) == 0.25)
            sim.frame(4)
            assert(sim.ticks() == 31, "quarter speed is a tick every four frames")
            assert(sim.speed(100) == 4, "clamped to the panel's range")
            assert(sim.speed() == 4)
            sim.frame()
            assert(sim.ticks() == 35)
            sim.step()
            assert(sim.ticks() == 36, "a step is one tick whatever the speed")
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
                "world.lua",
                r#"
                local names = sim.worlds()
                assert(names[1] == "Forest" and #names == 5, table.concat(names, ","))
                assert(sim.generate("forest", 1337) == 1337)
                assert(sim.count("Stone") > 50000)
                assert(sim.count("Wood") > 100, "trees")
                local seed = sim.generate("Desert")
                assert(seed >= 0)
                assert(sim.count("Water") == 0, "the desert is dry")
                assert(sim.count("Sand") > 5000)
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
            local brushes = sim.brushes()
            assert(brushes[1] == "Disk" and brushes[2] == "Spray", table.concat(brushes, ","))
            assert(sim.tools()[1] == "Fan")

            -- Three frames of the spray, so at least three grains land.
            sim.stroke { brush = "Spray", material = "Water", radius = 6,
                         path = { {100, 100}, {104, 100}, {108, 100} } }
            local world = sim.snapshot()
            local n = world:count("Water")
            assert(n >= 3, n)
            local x0, y0, x1, y1 = world:bounds("Water")
            assert(x0 >= 93 and x1 <= 115 and y0 >= 93 and y1 <= 107, "inside the brush")

            -- The panel's own pick is the default.
            sim.pick { material = "Stone", brush = "Disk", radius = 4 }
            sim.stroke { path = { {300, 300} } }
            assert(sim.count("Stone") > 40)

            -- A tool: the fan blows up, so the air above the cursor rises.
            sim.stroke { tool = "Fan", radius = 10, path = { {500, 400}, {500, 400}, {500, 400} } }
            local _, vy = sim.snapshot():wind(500, 390)
            assert(vy < 0, "the fan is an updraft: " .. vy)

            -- The wind tool needs a sweep; a single point blows nothing.
            sim.clear()
            sim.stroke { tool = "Wind", path = { {200, 250} } }
            local vx = sim.snapshot():wind(200, 250)
            assert(vx == 0, vx)
            sim.stroke { tool = "Wind", path = { {200, 250}, {210, 250} } }
            vx = sim.snapshot():wind(210, 250)
            assert(vx > 1, "a sweep to the right blows to the right: " .. vx)

            -- Picking a tool on the panel, then a stroke with no spec.
            sim.pick { tool = "wind" }
            sim.pick { tool = "Fan" }
        "#)
        .unwrap();

        for bad in [
            "sim.stroke { brush = 'Nope', path = {{1, 1}} }",
            "sim.stroke { tool = 'Nope', path = {{1, 1}} }",
            "sim.stroke { brush = 'Disk', tool = 'Fan', path = {{1, 1}} }",
            "sim.stroke { brush = 'Disk' }",
            "sim.stroke { brush = 'Disk', path = {{1}} }",
            "sim.pick { brush = 'Nope' }",
        ] {
            let err = run(bad).unwrap_err();
            assert!(err.contains("test.lua:1"), "{bad}: {err}");
        }
    }

    #[test]
    fn a_screenshot_is_written_before_the_call_returns() {
        let dir = std::env::temp_dir().join(format!("sandy-script-shot-{}", std::process::id()));
        let path = dir.join("shot.png");
        run(&format!(
            r#"
            sim.fill(0, 400, 999, 499, "Lava")
            local where = sim.screenshot({path:?})
            assert(where == {path:?}, where)
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
        run(&format!(
            r#"
            sim.paint(500, 100, 20, "Sand")
            assert(sim.record({path:?}) == {path:?})
            local ok, err = pcall(sim.record)
            assert(not ok and err:find("already"), err)
            sim.frame(20)
            sim.step(20)
            assert(sim.stop() == {path:?})
            "#
        ))
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
        run(&format!("sim.record({path:?}); sim.frame(4)")).unwrap();
        assert!(std::fs::read(&path).unwrap().starts_with(b"RIFF"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_script_can_load_a_plugin_and_use_what_it_added() {
        let dir = std::env::temp_dir().join(format!("sandy-script-plugin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plugin = dir.join("ash.lua");
        std::fs::write(
            &plugin,
            r#"
            sandy.material { name = "Ash", color = {90, 90, 90}, density = 60, mobile = true }
            sandy.brush { name = "Dot", on_drag = function(t) sandy.paint(t.x, t.y, 0, t.material) end }
            "#,
        )
        .unwrap();
        run(&format!(
            r#"
            local report = sim.plugin({plugin:?})
            assert(report:find("1 material") and report:find("1 brush"), report)
            local found
            for _, m in ipairs(sim.materials()) do
                if m.name == "Ash" then found = m end
            end
            assert(found and found.density == 60 and found.mobile and found.color[1] == 90)
            assert(found.id == sandy.find("Ash"))
            sim.stroke {{ brush = "Dot", material = "Ash", path = {{ {{10, 10}}, {{11, 10}} }} }}
            assert(sim.count("Ash") == 2)
            -- The registered material falls, so the tables reached the GPU.
            sim.step(30)
            assert(sim.snapshot():highest("Ash") > 20)
            -- A control script can register things itself, too.
            sandy.material {{ name = "Dust", color = {{1, 1, 1}}, density = 30, mobile = true }}
            "#
        ))
        .unwrap();
        let err = run(&format!("sim.plugin({:?})", dir.join("missing.lua"))).unwrap_err();
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
            "windowed.lua",
            r#"
            sim.paint(10, 10, 0, "Sand")
            sim.frame(2)
            sim.paint(20, 10, 0, "Sand")
            sim.frame()
            sim.quit()
            error("never reached")
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
    }
}
