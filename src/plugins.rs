//! Lua plugins: scripts that add materials, brushes, tools and worlds while
//! the game runs.
//!
//! A plugin is one `.lua` file. Dropped on the window, left in the `plugins`
//! folder next to where the game is run from, or compiled in as one of
//! [`BUILTIN`], it is executed once with a `sandy` table in scope, and whatever
//! it registers through that table is in the game from then on. A material is
//! a row in the [`Registry`] that [`crate::sim::Simulation::set_tables`] then
//! uploads, a brush or a tool is a Lua function the app calls every frame
//! the mouse is held with it selected, and a world is a Lua function that
//! paints a whole landscape when the panel asks for it.
//!
//! # What a script sees
//!
//! ```lua
//! local acid = sandy.material {
//!     name = "Acid", color = {120, 230, 60}, density = 120,
//!     mobile = true, liquid = true, spread = 200, glow = true,
//! }
//! sandy.rule { actor = "Stone", trigger = acid, product = "Empty",
//!              look = "around", chance = 6 }
//! sandy.brush { name = "Dot", on_drag = function(t)
//!     sandy.paint(t.x, t.y, 0, t.material)
//! end }
//! sandy.tool { name = "Fan", on_drag = function(t)
//!     sandy.wind(t.x, t.y, 30, 0, -4)
//! end }
//! sandy.world { name = "Flat", generate = function(w)
//!     local hills = sandy.noise { seed = w.seed, frequency = 0.01, octaves = 4 }
//!     for x = 0, w.width - 1 do
//!         local top = w.height * 0.6 - hills:at(x, 0) * 40
//!         w:fill(x, top, x, w.height - 1, "Soil")
//!     end
//! end }
//! sandy.find("Water")                   -- an id, or nil
//! sandy.paint(x, y, radius, material)   -- material is a name or an id
//! sandy.wind(x, y, radius, dvx, dvy)
//! sandy.noise { seed = 1, frequency = 0.01, octaves = 4 }
//! sandy.width, sandy.height             -- the grid, in cells
//! ```
//!
//! Materials, brushes, tools and worlds go by name. Registering a name that
//! already exists replaces the old entry in place, which is what makes
//! dropping a file on the window a second time a reload, and lets a plugin
//! retune a built-in material or brush.
//!
//! A brush and a tool are the same thing to this module, a function called
//! once a frame while the mouse is held (see [`Kind`]). The difference is what
//! the panel does with them: a brush paints the material picked in the panel,
//! in its own way, so a material and a brush are chosen together; a tool does
//! something else with the cursor. Even the plain brush is a script,
//! `disk.lua`, so there is one way to paint rather than a built-in way and a
//! plugin way.
//!
//! # Why a script never touches the world itself
//!
//! Materials are data the kernels read, so a script cannot give one an
//! `update` function any more than a Rust material can (see
//! [`crate::materials`]). What it can do is what the built-in materials do:
//! fill in the properties and the reaction rules. A tool is different. It runs
//! on the CPU once a frame, at cursor rate, so a real Lua function is fine
//! there. Even so, the function is not handed the simulation. `sandy.paint` and
//! `sandy.wind` queue a [`Command`], and the app drains the queue and applies
//! it once the script has returned, so the Lua state and the GPU state never
//! have to know about each other.
//!
//! A world is the same idea at a larger scale. Its `generate` function paints
//! into a [`Canvas`] in main memory, and when it returns the whole grid goes
//! to the GPU in one write (see [`Plugins::generate`]). The canvas is handed
//! to the script as an object, `w`, that only works while that one generation
//! is running.
//!
//! # Sandbox
//!
//! A script gets Lua's base library, `string`, `table`, `math`, `utf8` and
//! `coroutine`, and nothing that reaches outside the interpreter: no `io`, no
//! `os`, no `require`. `print` goes to the log. A plugin is something the user
//! chose to drop on the window, so this is not a security boundary, but a
//! plugin has no business with any of that and a broken one should not be able
//! to do much harm.

use std::cell::{Ref, RefCell};
use std::fmt;
use std::path::Path;
use std::rc::Rc;

use mlua::{
    FromLua, Function, Lua, LuaOptions, StdLib, Table, Thread, UserData, UserDataFields,
    UserDataMethods, Value, Variadic,
};

use crate::materials::{Look, MaterialId, MaterialInfo, Registry, Rule};
use crate::sim::{GRID_H, GRID_W, Simulation};
use crate::worldgen::{Canvas, Noise};

/// The widest brush a script can ask for, in cells. The paint and gust kernels
/// are dispatched over the brush's bounding square, so an unbounded radius
/// would be an unbounded amount of GPU work for one call.
const MAX_RADIUS: i32 = 512;

/// Where the user's own plugins are looked for at startup, relative to the
/// working directory. Every `.lua` file in it is loaded, in name order, after
/// the built-in ones.
pub const PLUGIN_DIR: &str = "plugins";

/// The plugins that ship with the game, compiled into the binary so they are
/// there wherever it is run from. They are ordinary scripts in `src/plugins/`
/// and go through the same loader as a dropped file, which also makes them the
/// worked examples of what a plugin can do.
pub const BUILTIN: &[(&str, &str)] = &[
    ("acid.lua", include_str!("plugins/acid.lua")),
    ("disk.lua", include_str!("plugins/disk.lua")),
    ("fan.lua", include_str!("plugins/fan.lua")),
    // Steam's rules name fire, so fire has to be registered first.
    ("fire.lua", include_str!("plugins/fire.lua")),
    ("spray.lua", include_str!("plugins/spray.lua")),
    ("steam.lua", include_str!("plugins/steam.lua")),
    // Wood's rules name fire too.
    ("wood.lua", include_str!("plugins/wood.lua")),
    // The worlds only name materials when they are generated, so they could
    // go anywhere, but the panel lists them in this order.
    ("worlds.lua", include_str!("plugins/worlds.lua")),
    ("caverns.lua", include_str!("plugins/caverns.lua")),
];

/// The two kinds of script a stroke can drive. They are registered with
/// `sandy.brush` and `sandy.tool` and kept in separate lists, so the panel can
/// put the brushes next to the materials and the tools on their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Paints the material picked in the panel, in its own way.
    Brush,
    /// Does something else with the cursor.
    Tool,
}

impl Kind {
    fn word(self) -> &'static str {
        match self {
            Kind::Brush => "brush",
            Kind::Tool => "tool",
        }
    }
}

/// Something a script asked the world to do, waiting for the app to do it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// Stamp a disk of a material, as the brush does.
    Paint {
        x: i32,
        y: i32,
        radius: i32,
        material: MaterialId,
    },
    /// Blow a gust, as the wind tool does. The push is in cells per tick.
    Wind {
        x: i32,
        y: i32,
        radius: i32,
        dvx: f32,
        dvy: f32,
    },
}

impl Command {
    /// Do it to the world.
    pub fn apply(self, sim: &mut Simulation) {
        match self {
            Command::Paint {
                x,
                y,
                radius,
                material,
            } => sim.paint_disk(x, y, radius, material),
            Command::Wind {
                x,
                y,
                radius,
                dvx,
                dvy,
            } => sim.add_wind_disk(x, y, radius, dvx, dvy),
        }
    }
}

/// One frame of a stroke, as a plugin tool sees it. Positions are grid cells.
#[derive(Clone, Copy, Debug)]
pub struct Stroke {
    pub x: i32,
    pub y: i32,
    /// Where the cursor was on the previous frame of this stroke, or the same
    /// place as `x`/`y` on its first frame, so `x - px` is always a direction.
    pub px: i32,
    pub py: i32,
    /// Whether this is the first frame since the button went down.
    pub first: bool,
    /// The size set in the panel, in cells.
    pub radius: i32,
    /// The material chosen in the panel, which is what a brush paints.
    pub material: MaterialId,
}

/// How much a script registered when it was loaded, for the panel to report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub materials: usize,
    pub rules: usize,
    pub brushes: usize,
    pub tools: usize,
    pub worlds: usize,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = |n: usize, one: &str, many: &str| {
            if n == 1 {
                format!("{n} {one}")
            } else {
                format!("{n} {many}")
            }
        };
        write!(
            f,
            "{}, {}, {}, {}, {}",
            count(self.materials, "material", "materials"),
            count(self.rules, "rule", "rules"),
            count(self.brushes, "brush", "brushes"),
            count(self.tools, "tool", "tools"),
            count(self.worlds, "world", "worlds")
        )
    }
}

/// A brush or a tool a script registered.
struct PluginTool {
    name: String,
    on_drag: Function,
}

/// A world preset a script registered: a name for the panel and the function
/// that paints it.
struct PluginWorld {
    name: String,
    generate: Function,
}

/// Everything the `sandy` functions write to. The Lua closures each hold a
/// handle to this, and so does [`Plugins`], which is how what a script did
/// gets back out.
struct Shared {
    registry: Registry,
    brushes: Vec<PluginTool>,
    tools: Vec<PluginTool>,
    worlds: Vec<PluginWorld>,
    commands: Vec<Command>,
    /// The world being generated, while a world's `generate` function is
    /// running, and nothing the rest of the time. See [`Plugins::generate`].
    canvas: Option<Canvas>,
    /// Which generation is running, counted up each time one starts, so a
    /// handle a script kept from an earlier one can be told apart from the
    /// current one.
    generation: u64,
    /// What the script currently being loaded has registered so far.
    report: Report,
}

impl Shared {
    fn list(&self, kind: Kind) -> &[PluginTool] {
        match kind {
            Kind::Brush => &self.brushes,
            Kind::Tool => &self.tools,
        }
    }

    fn list_mut(&mut self, kind: Kind) -> &mut Vec<PluginTool> {
        match kind {
            Kind::Brush => &mut self.brushes,
            Kind::Tool => &mut self.tools,
        }
    }
}

/// `sandy.brush` and `sandy.tool`: the same registration, into different
/// lists. A name already in the list is replaced in place, so its position,
/// which is what the panel's selection refers to, stays put.
fn register(shared: &Rc<RefCell<Shared>>, kind: Kind, spec: Table) -> mlua::Result<()> {
    let name: String = required(&spec, "name")?;
    if name.trim().is_empty() {
        return Err(runtime(format!("a {} needs a name", kind.word())));
    }
    let on_drag: Function = required(&spec, "on_drag")?;
    let mut shared = shared.borrow_mut();
    let list = shared.list_mut(kind);
    match list
        .iter_mut()
        .find(|entry| entry.name.eq_ignore_ascii_case(&name))
    {
        Some(entry) => entry.on_drag = on_drag,
        None => list.push(PluginTool { name, on_drag }),
    }
    match kind {
        Kind::Brush => shared.report.brushes += 1,
        Kind::Tool => shared.report.tools += 1,
    }
    Ok(())
}

/// `sandy.world`: the same registration again, for a landscape. A name
/// already in the list is replaced in place, so the panel's choice of world
/// still points at the same entry.
fn register_world(shared: &Rc<RefCell<Shared>>, spec: Table) -> mlua::Result<()> {
    let name: String = required(&spec, "name")?;
    if name.trim().is_empty() {
        return Err(runtime("a world needs a name"));
    }
    let generate: Function = required(&spec, "generate")?;
    let mut shared = shared.borrow_mut();
    match shared
        .worlds
        .iter_mut()
        .find(|entry| entry.name.eq_ignore_ascii_case(&name))
    {
        Some(entry) => entry.generate = generate,
        None => shared.worlds.push(PluginWorld { name, generate }),
    }
    shared.report.worlds += 1;
    Ok(())
}

/// What a world's `generate` function is handed: the seed and the size of
/// the world, and the methods that paint into the canvas. It is only good
/// for the one generation it was made for; the canvas is taken away when the
/// function returns, and a stashed handle used later says so.
struct World {
    shared: Rc<RefCell<Shared>>,
    seed: u32,
    /// The generation this handle belongs to; see [`Shared::generation`].
    generation: u64,
}

impl World {
    /// The canvas, as long as this handle's generation is the one running.
    fn canvas<'a>(&self, shared: &'a mut Shared) -> mlua::Result<&'a mut Canvas> {
        if shared.generation != self.generation {
            return Err(runtime("this world has already been built"));
        }
        shared
            .canvas
            .as_mut()
            .ok_or_else(|| runtime("this world has already been built"))
    }

    /// Run `f` on the canvas, with a material a script named resolved to its
    /// id first, since both live in the same borrow.
    fn paint(&self, material: &Value, f: impl FnOnce(&mut Canvas, MaterialId)) -> mlua::Result<()> {
        let shared = &mut *self.shared.borrow_mut();
        let material = resolve(&shared.registry, material, "material")?;
        let canvas = self.canvas(shared)?;
        f(canvas, material);
        Ok(())
    }
}

/// A cell coordinate a script gave, which may be a float from its arithmetic.
/// Rounded down, so a tree planted at `top - 0.5` stands on the cell above.
fn cell(v: f64) -> i64 {
    if v.is_finite() {
        v.floor() as i64
    } else {
        // Off the grid, so anything painted here is clipped away.
        i64::MIN / 2
    }
}

impl UserData for World {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("seed", |_, w| Ok(w.seed));
        fields.add_field_method_get("width", |_, _| Ok(GRID_W));
        fields.add_field_method_get("height", |_, _| Ok(GRID_H));
    }

    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("set", |_, w, (x, y, material): (f64, f64, Value)| {
            w.paint(&material, |canvas, m| canvas.set(cell(x), cell(y), m))
        });
        methods.add_method("get", |_, w, (x, y): (f64, f64)| {
            let shared = &mut *w.shared.borrow_mut();
            Ok(w.canvas(shared)?.get(cell(x), cell(y)))
        });
        methods.add_method(
            "fill",
            |_, w, (x0, y0, x1, y1, material): (f64, f64, f64, f64, Value)| {
                w.paint(&material, |canvas, m| {
                    canvas.fill(cell(x0), cell(y0), cell(x1), cell(y1), m)
                })
            },
        );
        methods.add_method(
            "disk",
            |_, w, (x, y, radius, material): (f64, f64, f64, Value)| {
                let radius = clamp_radius(radius) as i64;
                w.paint(&material, |canvas, m| {
                    canvas.disk(cell(x), cell(y), radius, m)
                })
            },
        );
    }
}

/// The Lua interpreter, the materials, brushes and tools the scripts have
/// registered, and the commands they have queued.
pub struct Plugins {
    lua: Lua,
    shared: Rc<RefCell<Shared>>,
}

impl Default for Plugins {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugins {
    /// An interpreter with the `sandy` table in it and the built-in materials
    /// in its registry, and no scripts loaded yet.
    pub fn new() -> Self {
        Self::build().expect("build the Lua interpreter")
    }

    fn build() -> mlua::Result<Self> {
        let lua = Lua::new_with(
            StdLib::COROUTINE | StdLib::TABLE | StdLib::STRING | StdLib::UTF8 | StdLib::MATH,
            LuaOptions::default(),
        )?;
        let shared = Rc::new(RefCell::new(Shared {
            registry: Registry::builtin(),
            brushes: Vec::new(),
            tools: Vec::new(),
            worlds: Vec::new(),
            commands: Vec::new(),
            canvas: None,
            generation: 0,
            report: Report::default(),
        }));

        // `print` would otherwise go to stdout, which nobody is watching.
        lua.globals().set(
            "print",
            lua.create_function(|_, args: Variadic<Value>| {
                let parts: mlua::Result<Vec<String>> = args.iter().map(|v| v.to_string()).collect();
                log::info!(target: "plugin", "{}", parts?.join("\t"));
                Ok(())
            })?,
        )?;

        let sandy = lua.create_table()?;
        sandy.set("width", GRID_W)?;
        sandy.set("height", GRID_H)?;

        let s = shared.clone();
        sandy.set(
            "material",
            lua.create_function(move |_, spec: Table| {
                let info = material_from(&spec)?;
                let mut shared = s.borrow_mut();
                let id = shared
                    .registry
                    .add_material(info)
                    .map_err(mlua::Error::runtime)?;
                shared.report.materials += 1;
                Ok(id)
            })?,
        )?;

        let s = shared.clone();
        sandy.set(
            "rule",
            lua.create_function(move |_, spec: Table| {
                let shared = &mut *s.borrow_mut();
                let actor = resolve(&shared.registry, &spec.get("actor")?, "actor")?;
                let trigger = resolve(&shared.registry, &spec.get("trigger")?, "trigger")?;
                let product = resolve(&shared.registry, &spec.get("product")?, "product")?;
                let look = look_from(optional(&spec, "look", None)?)?;
                let chance: u32 = optional(&spec, "chance", 1)?;
                if chance == 0 {
                    return Err(runtime(
                        "chance is one in how many ticks, so it cannot be 0",
                    ));
                }
                shared.registry.add_rule(Rule {
                    actor,
                    trigger,
                    product,
                    look,
                    chance,
                });
                shared.report.rules += 1;
                Ok(())
            })?,
        )?;

        for (key, kind) in [("brush", Kind::Brush), ("tool", Kind::Tool)] {
            let s = shared.clone();
            sandy.set(
                key,
                lua.create_function(move |_, spec: Table| register(&s, kind, spec))?,
            )?;
        }

        let s = shared.clone();
        sandy.set(
            "world",
            lua.create_function(move |_, spec: Table| register_world(&s, spec))?,
        )?;

        sandy.set(
            "noise",
            lua.create_function(|_, spec: Table| Noise::from_spec(&spec))?,
        )?;

        let s = shared.clone();
        sandy.set(
            "find",
            lua.create_function(move |_, name: String| Ok(s.borrow().registry.find(&name)))?,
        )?;

        let s = shared.clone();
        sandy.set(
            "paint",
            lua.create_function(move |_, (x, y, radius, material): (f64, f64, f64, Value)| {
                let mut shared = s.borrow_mut();
                let material = resolve(&shared.registry, &material, "material")?;
                shared.commands.push(Command::Paint {
                    x: x.round() as i32,
                    y: y.round() as i32,
                    radius: clamp_radius(radius),
                    material,
                });
                Ok(())
            })?,
        )?;

        let s = shared.clone();
        sandy.set(
            "wind",
            lua.create_function(
                move |_, (x, y, radius, dvx, dvy): (f64, f64, f64, f64, f64)| {
                    s.borrow_mut().commands.push(Command::Wind {
                        x: x.round() as i32,
                        y: y.round() as i32,
                        radius: clamp_radius(radius),
                        dvx: dvx as f32,
                        dvy: dvy as f32,
                    });
                    Ok(())
                },
            )?,
        )?;

        lua.globals().set("sandy", sandy)?;
        Ok(Plugins { lua, shared })
    }

    /// Run every script in [`BUILTIN`]. One failing is a bug in the repo rather
    /// than in anything the user did, so it is logged and the rest still load.
    pub fn load_builtin(&mut self) {
        for (name, source) in BUILTIN {
            match self.load(name, source) {
                Ok(report) => log::info!("loaded built-in plugin {name}: {report}"),
                Err(err) => log::error!("built-in plugin {name} failed: {err}"),
            }
        }
    }

    /// Load every `.lua` file in [`PLUGIN_DIR`], in name order, and say how
    /// each went: its file name, and the report or the error. No folder is
    /// simply no plugins; a script that fails does not stop the rest loading.
    pub fn load_dir(&mut self) -> Vec<(String, Result<Report, String>)> {
        let Ok(entries) = std::fs::read_dir(PLUGIN_DIR) else {
            return Vec::new();
        };
        let mut paths: Vec<_> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "lua"))
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|path| {
                let label = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                let result = self.load_file(path);
                match &result {
                    Ok(report) => log::info!("loaded plugin {}: {report}", path.display()),
                    Err(err) => log::error!("plugin {} failed: {err}", path.display()),
                }
                (label, result)
            })
            .collect()
    }

    /// Run the script in `path`. What it registers stays registered even if it
    /// then fails partway through, so a script that got as far as adding a
    /// material has added it.
    pub fn load_file(&mut self, path: &Path) -> Result<Report, String> {
        let source =
            std::fs::read_to_string(path).map_err(|err| format!("could not read it: {err}"))?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.load(&name, &source)
    }

    /// Run `source` as a script. `name` is what Lua puts in front of the line
    /// number in an error message.
    pub fn load(&mut self, name: &str, source: &str) -> Result<Report, String> {
        self.shared.borrow_mut().report = Report::default();
        // The `=` tells Lua to use the name as it is. Without it the name is
        // taken for source text and shows up as `[string "acid.lua"]`.
        let result = self.lua.load(source).set_name(format!("={name}")).exec();
        let report = self.shared.borrow().report;
        result.map(|()| report).map_err(|err| describe(&err))
    }

    /// The interpreter itself, for the control script in [`crate::scripting`],
    /// which runs in it alongside the plugins.
    pub(crate) fn lua(&self) -> &Lua {
        &self.lua
    }

    /// `source` compiled as a coroutine, not yet started, named `name` in
    /// error messages the way [`Plugins::load`] names a plugin. A syntax
    /// error shows up here rather than on the first resume.
    pub(crate) fn thread(&self, name: &str, source: &str) -> Result<Thread, String> {
        let function = self
            .lua
            .load(source)
            .set_name(format!("={name}"))
            .into_function()
            .map_err(|err| describe(&err))?;
        self.lua
            .create_thread(function)
            .map_err(|err| describe(&err))
    }

    /// The names of the brushes or tools scripts have registered, in the order
    /// they were added. The panel's choice of brush, and a
    /// [`crate::ui::Tool::Plugin`], are indexes into these.
    pub fn names(&self, kind: Kind) -> Vec<String> {
        self.shared
            .borrow()
            .list(kind)
            .iter()
            .map(|entry| entry.name.clone())
            .collect()
    }

    /// Where the brush or tool called `name`, in any case, sits in
    /// [`Plugins::names`].
    pub fn index_of(&self, kind: Kind, name: &str) -> Option<usize> {
        self.shared
            .borrow()
            .list(kind)
            .iter()
            .position(|entry| entry.name.eq_ignore_ascii_case(name))
    }

    /// Where the world called `name`, in any case, sits in
    /// [`Plugins::world_names`].
    pub fn world_index(&self, name: &str) -> Option<usize> {
        self.shared
            .borrow()
            .worlds
            .iter()
            .position(|entry| entry.name.eq_ignore_ascii_case(name))
    }

    /// Give a brush or a tool one frame of a stroke. Whatever it asks for lands
    /// in the command queue; see [`Plugins::take_commands`].
    pub fn run(&mut self, kind: Kind, index: usize, stroke: Stroke) -> Result<(), String> {
        // The function is cloned out so the borrow is released before the
        // call, since the script will want to borrow the queue itself.
        let on_drag = self
            .shared
            .borrow()
            .list(kind)
            .get(index)
            .map(|entry| entry.on_drag.clone())
            .ok_or_else(|| format!("there is no {} number {index}", kind.word()))?;
        let frame = (|| {
            let t = self.lua.create_table()?;
            t.set("x", stroke.x)?;
            t.set("y", stroke.y)?;
            t.set("px", stroke.px)?;
            t.set("py", stroke.py)?;
            t.set("first", stroke.first)?;
            t.set("radius", stroke.radius)?;
            t.set("material", stroke.material)?;
            Ok(t)
        })()
        .map_err(|err: mlua::Error| describe(&err))?;
        on_drag.call::<()>(frame).map_err(|err| describe(&err))
    }

    /// The names of the worlds scripts have registered, in the order they were
    /// added. The panel's choice of world is an index into these.
    pub fn world_names(&self) -> Vec<String> {
        self.shared
            .borrow()
            .worlds
            .iter()
            .map(|entry| entry.name.clone())
            .collect()
    }

    /// Build world number `index` from `seed`: run its `generate` function
    /// over a fresh canvas and hand back the grid it painted, one material
    /// per cell from the top left, for [`crate::sim::Simulation::load`].
    ///
    /// The same world and seed always give the same grid. Lua's `math.random`
    /// is reseeded from `seed` first, so a script can scatter trees with it
    /// and still be reproducible; the noise it makes carries its own seed.
    pub fn generate(&mut self, index: usize, seed: u32) -> Result<Vec<MaterialId>, String> {
        let generate = self
            .shared
            .borrow()
            .worlds
            .get(index)
            .map(|entry| entry.generate.clone())
            .ok_or_else(|| format!("there is no world number {index}"))?;
        let generation = {
            let mut shared = self.shared.borrow_mut();
            shared.canvas = Some(Canvas::new(GRID_W, GRID_H));
            shared.generation += 1;
            shared.generation
        };

        let result = (|| {
            let math: Table = self.lua.globals().get("math")?;
            math.get::<Function>("randomseed")?.call::<()>(seed)?;
            let world = World {
                shared: self.shared.clone(),
                seed,
                generation,
            };
            generate.call::<()>(world)
        })();
        // The canvas comes out whatever happened, so a script that failed
        // halfway leaves nothing behind for the next one to paint over.
        let canvas = self.shared.borrow_mut().canvas.take();
        result.map_err(|err| describe(&err))?;
        Ok(canvas.expect("the canvas is only taken here").into_cells())
    }

    /// Every material and rule, built in and from scripts.
    pub fn registry(&self) -> Ref<'_, Registry> {
        Ref::map(self.shared.borrow(), |shared| &shared.registry)
    }

    /// Everything scripts have asked the world to do since this was last
    /// called, in order.
    pub fn take_commands(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.shared.borrow_mut().commands)
    }
}

pub(crate) fn runtime(message: impl fmt::Display) -> mlua::Error {
    mlua::Error::runtime(message.to_string())
}

/// A field that has to be there.
fn required<T: FromLua>(spec: &Table, key: &str) -> mlua::Result<T> {
    optional(spec, key, None)?.ok_or_else(|| runtime(format!("'{key}' is required")))
}

/// A field that may be missing, in which case `default` stands in.
pub(crate) fn optional<T: FromLua>(spec: &Table, key: &str, default: T) -> mlua::Result<T> {
    let value: Option<T> = spec
        .get(key)
        .map_err(|err| runtime(format!("'{key}': {}", plain(&err))))?;
    Ok(value.unwrap_or(default))
}

fn material_from(spec: &Table) -> mlua::Result<MaterialInfo> {
    let name: String = required(spec, "name")?;
    if name.trim().is_empty() {
        return Err(runtime("a material needs a name"));
    }
    let color: Vec<u8> = required(spec, "color")?;
    let [r, g, b] = color[..] else {
        return Err(runtime(
            "'color' should be three numbers from 0 to 255, like {200, 120, 40}",
        ));
    };
    Ok(MaterialInfo {
        // Material names are `&'static str` because the built-in ones are
        // constants. A script's material lives as long as the game does too,
        // so its name is simply never freed. Reloading a script leaks the few
        // bytes of the old name, which is nothing.
        name: name.leak(),
        color: [r, g, b],
        jitter: optional(spec, "jitter", 0)?,
        density: required(spec, "density")?,
        mobile: optional(spec, "mobile", false)?,
        passable: optional(spec, "passable", true)?,
        liquid: optional(spec, "liquid", false)?,
        spread: optional(spec, "spread", 0)?,
        windborne: optional(spec, "windborne", false)?,
        glow: optional(spec, "glow", false)?,
        draft: optional(spec, "draft", 0)?,
    })
}

/// A material named by a script: its id, or its name in any case.
pub(crate) fn resolve(registry: &Registry, value: &Value, what: &str) -> mlua::Result<MaterialId> {
    let by_id = |id: i64| {
        if (0..registry.materials().len() as i64).contains(&id) {
            Ok(id as MaterialId)
        } else {
            Err(runtime(format!(
                "{what}: there is no material with id {id}"
            )))
        }
    };
    match value {
        Value::Integer(id) => by_id(*id),
        Value::Number(n) if n.fract() == 0.0 => by_id(*n as i64),
        Value::String(name) => {
            let name = name.to_str()?;
            registry
                .find(&name)
                .ok_or_else(|| runtime(format!("{what}: there is no material called '{}'", &*name)))
        }
        Value::Nil => Err(runtime(format!("'{what}' is required"))),
        other => Err(runtime(format!(
            "{what} should be a material name or id, not a {}",
            other.type_name()
        ))),
    }
}

fn look_from(name: Option<String>) -> mlua::Result<Look> {
    match name.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("ortho") => Ok(Look::Ortho),
        Some("around") => Ok(Look::Around),
        Some("above") => Ok(Look::Above),
        Some("below") => Ok(Look::Below),
        Some(other) => Err(runtime(format!(
            "look should be ortho, around, above or below, not '{other}'"
        ))),
    }
}

pub(crate) fn clamp_radius(radius: f64) -> i32 {
    (radius.round() as i32).clamp(0, MAX_RADIUS)
}

/// The one line of an error worth showing on the panel: what went wrong, and
/// where in the script if that can be told. Lua's own errors say where on
/// their first line; an error raised from the Rust side does not, but the
/// traceback under it does, so that is looked for and added. Frames in C
/// and in the control API's own Lua are passed over, since neither is
/// anywhere the user wrote.
pub(crate) fn describe(err: &mlua::Error) -> String {
    let text = err.to_string();
    let first = plain(err);
    let prelude = format!("{}:", crate::scripting::PRELUDE_NAME);
    let place = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("[C]") && !line.starts_with(&prelude))
        .find_map(|line| line.split_once(": in ").map(|(place, _)| place));
    match place {
        Some(place) if !first.starts_with(place) => format!("{first} (at {place})"),
        _ => first,
    }
}

/// An error's message with mlua's own framing taken off: the root cause of a
/// callback error, and only the first line of it.
pub(crate) fn plain(err: &mlua::Error) -> String {
    let mut cause = err;
    while let mlua::Error::CallbackError { cause: inner, .. } = cause {
        cause = inner;
    }
    let text = cause.to_string();
    let first = text.lines().next().unwrap_or("unknown error").trim();
    first
        .strip_prefix("runtime error: ")
        .or_else(|| first.strip_prefix("syntax error: "))
        .unwrap_or(first)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{EMPTY, LAVA, SAND, SOIL, STONE, WATER};

    #[test]
    fn a_script_can_add_a_material_and_a_rule() {
        let mut plugins = Plugins::new();
        let report = plugins
            .load(
                "acid.lua",
                r#"
                local acid = sandy.material {
                    name = "Acid", color = {1, 2, 3}, density = 120,
                    mobile = true, liquid = true, spread = 200, glow = true,
                }
                sandy.rule { actor = "stone", trigger = acid, product = "Empty",
                             look = "around", chance = 6 }
                "#,
            )
            .unwrap();
        assert_eq!(
            report,
            Report {
                materials: 1,
                rules: 1,
                brushes: 0,
                tools: 0,
                worlds: 0,
            }
        );
        assert_eq!(
            report.to_string(),
            "1 material, 1 rule, 0 brushes, 0 tools, 0 worlds"
        );

        let registry = plugins.registry();
        let acid = registry.find("Acid").expect("the material was registered");
        assert_eq!(acid as usize, registry.materials().len() - 1);
        let info = &registry.materials()[acid as usize];
        assert_eq!(info.name, "Acid");
        assert_eq!(info.color, [1, 2, 3]);
        assert_eq!(info.density, 120);
        assert!(info.mobile && info.liquid && info.glow);
        assert!(info.passable, "passable defaults to true");
        assert!(!info.windborne, "windborne defaults to false");
        assert_eq!(info.jitter, 0);
        assert_eq!(info.draft, 0, "draft defaults to zero");

        let rule = registry
            .rules()
            .iter()
            .find(|rule| rule.actor == STONE && rule.trigger == acid)
            .expect("the rule was registered");
        assert_eq!(rule.product, EMPTY);
        assert_eq!(rule.look, Look::Around);
        assert_eq!(rule.chance, 6);
    }

    #[test]
    fn loading_a_script_again_replaces_rather_than_duplicates() {
        let mut plugins = Plugins::new();
        let script = |color: &str| {
            format!(
                r#"sandy.material {{ name = "Ash", color = {color}, density = 100, mobile = true }}
                   sandy.tool {{ name = "Puff", on_drag = function(t) end }}"#
            )
        };
        plugins.load("ash.lua", &script("{9, 9, 9}")).unwrap();
        let count = plugins.registry().materials().len();
        let id = plugins.registry().find("Ash").unwrap();

        plugins.load("ash.lua", &script("{4, 4, 4}")).unwrap();
        let registry = plugins.registry();
        assert_eq!(registry.materials().len(), count);
        assert_eq!(registry.find("Ash"), Some(id));
        assert_eq!(registry.materials()[id as usize].color, [4, 4, 4]);
        assert_eq!(plugins.names(Kind::Tool), ["Puff"]);
    }

    #[test]
    fn a_tool_queues_commands_for_the_app_to_run() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "fan.lua",
                r#"
                sandy.tool { name = "Fan", on_drag = function(t)
                    sandy.wind(t.x, t.y, t.radius * 3, 0, -4)
                    if t.first then sandy.paint(t.x, t.y, 2.4, t.material) end
                    if t.x ~= t.px then sandy.paint(t.px, t.py, 0, "water") end
                end }
                "#,
            )
            .unwrap();
        assert_eq!(plugins.names(Kind::Tool), ["Fan"]);
        assert!(plugins.names(Kind::Brush).is_empty());
        assert!(plugins.take_commands().is_empty());

        let stroke = Stroke {
            x: 10,
            y: 20,
            px: 10,
            py: 20,
            first: true,
            radius: 5,
            material: SAND,
        };
        plugins.run(Kind::Tool, 0, stroke).unwrap();
        assert_eq!(
            plugins.take_commands(),
            [
                Command::Wind {
                    x: 10,
                    y: 20,
                    radius: 15,
                    dvx: 0.0,
                    dvy: -4.0
                },
                Command::Paint {
                    x: 10,
                    y: 20,
                    radius: 2,
                    material: SAND
                },
            ]
        );

        plugins
            .run(
                Kind::Tool,
                0,
                Stroke {
                    x: 12,
                    px: 10,
                    first: false,
                    ..stroke
                },
            )
            .unwrap();
        assert_eq!(
            plugins.take_commands(),
            [
                Command::Wind {
                    x: 12,
                    y: 20,
                    radius: 15,
                    dvx: 0.0,
                    dvy: -4.0
                },
                Command::Paint {
                    x: 10,
                    y: 20,
                    radius: 0,
                    material: WATER
                },
            ]
        );
        assert!(
            plugins.take_commands().is_empty(),
            "taking the queue empties it"
        );
    }

    #[test]
    fn a_broken_script_reports_what_went_wrong_and_the_interpreter_carries_on() {
        let mut plugins = Plugins::new();

        let err = plugins.load("bad.lua", "sandy.material {").unwrap_err();
        assert!(err.contains("bad.lua"), "a syntax error says where: {err}");

        let err = plugins
            .load(
                "bad.lua",
                r#"sandy.material { name = "Ash", color = {1, 2, 3} }"#,
            )
            .unwrap_err();
        assert!(err.contains("'density' is required"), "{err}");
        assert!(
            err.contains("bad.lua:1"),
            "a callback error says where: {err}"
        );

        let err = plugins
            .load(
                "bad.lua",
                r#"sandy.rule { actor = "Sand", trigger = "Unobtainium", product = 0 }"#,
            )
            .unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");

        let err = plugins
            .load("bad.lua", r#"sandy.rule { actor = "Sand", trigger = "Water", product = 0, look = "sideways" }"#)
            .unwrap_err();
        assert!(err.contains("sideways"), "{err}");

        let err = plugins
            .load(
                "bad.lua",
                r#"sandy.material { name = "Ash", color = {1, 2}, density = 5 }"#,
            )
            .unwrap_err();
        assert!(err.contains("color"), "{err}");

        let err = plugins.load("bad.lua", "error('no thanks')").unwrap_err();
        assert!(err.contains("no thanks"), "{err}");
        assert!(err.contains("bad.lua:1"), "{err}");

        plugins
            .load(
                "tool.lua",
                "sandy.tool { name = 'Oops', on_drag = function(t) return t.nothing.here end }",
            )
            .unwrap();
        let stroke = Stroke {
            x: 0,
            y: 0,
            px: 0,
            py: 0,
            first: true,
            radius: 1,
            material: SAND,
        };
        let err = plugins.run(Kind::Tool, 0, stroke).unwrap_err();
        assert!(err.contains("tool.lua:1"), "{err}");
        assert!(plugins.run(Kind::Tool, 1, stroke).is_err(), "no such tool");
        assert!(
            plugins.run(Kind::Brush, 0, stroke).is_err(),
            "no brushes at all"
        );

        // None of that has hurt the interpreter or the registry.
        let registry_size = plugins.registry().materials().len();
        assert_eq!(plugins.registry().find("Ash"), None);
        plugins
            .load(
                "good.lua",
                r#"sandy.material { name = "Ash", color = {1, 2, 3}, density = 5 }"#,
            )
            .unwrap();
        assert_eq!(plugins.registry().materials().len(), registry_size + 1);
    }

    #[test]
    fn the_built_in_plugins_load_cleanly() {
        let mut plugins = Plugins::new();
        for (name, source) in BUILTIN {
            plugins
                .load(name, source)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
        }
        let registry = plugins.registry();
        let acid = registry.find("Acid").expect("acid.lua adds Acid");
        assert!(registry.materials()[acid as usize].liquid);
        assert!(
            registry.rules().iter().any(|rule| rule.trigger == acid),
            "acid.lua adds rules that fire next to acid"
        );
        drop(registry);
        assert_eq!(plugins.names(Kind::Brush), ["Disk", "Spray"]);
        assert_eq!(plugins.names(Kind::Tool), ["Fan"]);
        assert_eq!(
            plugins.world_names(),
            ["Forest", "Plains", "Ocean", "Desert", "Caverns"]
        );

        // The plain brush is brush zero, the panel's default, and it paints
        // the chosen material at the chosen size, which is all it does.
        let stroke = Stroke {
            x: 40,
            y: 50,
            px: 40,
            py: 50,
            first: true,
            radius: 8,
            material: WATER,
        };
        plugins.run(Kind::Brush, 0, stroke).unwrap();
        assert_eq!(
            plugins.take_commands(),
            [Command::Paint {
                x: 40,
                y: 50,
                radius: 8,
                material: WATER
            }]
        );

        // The spray puts down a scatter of single cells, all within the brush.
        plugins.run(Kind::Brush, 1, stroke).unwrap();
        let commands = plugins.take_commands();
        assert!(commands.len() > 1);
        for command in commands {
            let Command::Paint {
                x,
                y,
                radius,
                material,
            } = command
            else {
                panic!("the spray only paints, but queued {command:?}");
            };
            assert_eq!((radius, material), (0, WATER));
            assert!(
                (x - 40).pow(2) + (y - 50).pow(2) <= 9 * 9,
                "({x}, {y}) is outside the brush"
            );
        }
    }

    /// How many cells of each material a generated world holds.
    fn census(cells: &[MaterialId]) -> impl Fn(MaterialId) -> usize + '_ {
        move |material| cells.iter().filter(|&&m| m == material).count()
    }

    #[test]
    fn a_script_can_register_a_world_and_paint_it() {
        let mut plugins = Plugins::new();
        let report = plugins
            .load(
                "flat.lua",
                r#"
                sandy.world { name = "Flat", generate = function(w)
                    assert(w.width == sandy.width and w.height == sandy.height)
                    assert(w:get(0, 0) == 0, "a fresh canvas is air")
                    assert(w:get(-1, 0) == nil, "off the grid is nil")
                    -- A floor, a pond in it, a boulder, and one grain of sand
                    -- off the edge that goes nowhere.
                    w:fill(0, w.height - 10, w.width - 1, w.height - 1, "Stone")
                    w:fill(100, w.height - 10, 199, w.height - 6, "water")
                    w:disk(500, 100, 3, sandy.find("Soil"))
                    w:set(w.width, 5, "Sand")
                    assert(w:get(500, 100) == sandy.find("Soil"))
                end }
                "#,
            )
            .unwrap();
        assert_eq!(report.worlds, 1);
        assert_eq!(plugins.world_names(), ["Flat"]);

        let cells = plugins.generate(0, 5).unwrap();
        assert_eq!(cells.len(), (GRID_W * GRID_H) as usize);
        let count = census(&cells);
        assert_eq!(count(WATER), 100 * 5);
        assert_eq!(count(STONE), (GRID_W * 10) as usize - 100 * 5);
        assert_eq!(count(SAND), 0, "off the grid is clipped");
        assert_eq!(count(SOIL), 29, "a disk of radius three");
        assert_eq!(cells[(100 * GRID_W + 500) as usize], SOIL);

        // Registering the same name again replaces the world in place.
        plugins
            .load(
                "flat.lua",
                r#"sandy.world { name = "flat", generate = function(w) end }"#,
            )
            .unwrap();
        assert_eq!(plugins.world_names(), ["Flat"]);
        assert!(plugins.generate(0, 5).unwrap().iter().all(|&m| m == EMPTY));
        assert!(plugins.generate(1, 5).is_err(), "no such world");
    }

    #[test]
    fn the_same_seed_builds_the_same_world_and_another_seed_a_different_one() {
        let mut plugins = Plugins::new();
        plugins.load_builtin();
        let forest = plugins
            .world_names()
            .iter()
            .position(|name| name == "Forest")
            .unwrap();
        let first = plugins.generate(forest, 1337).unwrap();
        let again = plugins.generate(forest, 1337).unwrap();
        let other = plugins.generate(forest, 1338).unwrap();
        assert_eq!(first, again, "the seed decides everything, trees included");
        assert_ne!(first, other);

        // The forest has ground, water in the valleys, and trees.
        let count = census(&first);
        let registry = plugins.registry();
        let wood = registry.find("Wood").unwrap();
        let leaves = registry.find("Leaves").unwrap();
        assert!(count(SOIL) > 5_000);
        assert!(count(STONE) > 50_000);
        assert!(count(WATER) > 3_000, "only {} water", count(WATER));
        assert!(count(wood) > 100, "only {} wood", count(wood));
        assert!(
            count(leaves) > count(wood),
            "canopies are bigger than trunks"
        );
        assert!(count(EMPTY) > 100_000, "there is sky");
        // The top row is sky and the bottom row is rock, whatever the seed.
        assert!(first[..GRID_W as usize].iter().all(|&m| m == EMPTY));
        assert!(
            first[((GRID_H - 1) * GRID_W) as usize..]
                .iter()
                .all(|&m| m == STONE)
        );
    }

    #[test]
    fn every_built_in_world_builds_and_is_its_own_kind_of_place() {
        let mut plugins = Plugins::new();
        plugins.load_builtin();
        let names = plugins.world_names();
        let by_name = |name: &str| names.iter().position(|n| n == name).unwrap();
        let build =
            |plugins: &mut Plugins, name: &str| plugins.generate(by_name(name), 99).unwrap();

        let plains = build(&mut plugins, "Plains");
        assert_eq!(census(&plains)(WATER), 0, "the plains are dry");
        assert!(census(&plains)(SOIL) > 5_000);

        let ocean = build(&mut plugins, "Ocean");
        let count = census(&ocean);
        assert!(count(WATER) > 300_000, "the ocean is mostly water");
        assert!(count(SAND) > 10_000, "over a sandy bed");
        assert_eq!(count(SOIL) + count(STONE), 0);

        let desert = build(&mut plugins, "Desert");
        let count = census(&desert);
        assert_eq!(count(WATER), 0);
        assert!(count(SAND) > 5_000 && count(STONE) > 50_000);

        let caverns = build(&mut plugins, "Caverns");
        let count = census(&caverns);
        assert!(count(STONE) > 150_000, "mostly rock");
        assert!(count(EMPTY) > 120_000, "with the sky and the hollows");
        assert!(
            count(WATER) > 0 && count(LAVA) > 0,
            "pools in the deep pockets"
        );
    }

    #[test]
    fn a_world_that_fails_says_where_and_leaves_nothing_behind() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "bad.lua",
                r#"
                stash = nil
                sandy.world { name = "Half", generate = function(w)
                    stash = w
                    w:fill(0, 0, 10, 10, "Stone")
                    w:set(0, 0, "Unobtainium")
                end }
                sandy.world { name = "Late", generate = function(w)
                    return stash:get(0, 0)
                end }
                "#,
            )
            .unwrap();
        let err = plugins.generate(0, 1).unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");
        assert!(err.contains("bad.lua:6"), "{err}");

        // The canvas the failed script painted on is gone, and so a handle to
        // it kept from that run is no use to a later one.
        let err = plugins.generate(1, 1).unwrap_err();
        assert!(err.contains("already been built"), "{err}");

        // A world with no generate function is refused at registration.
        let err = plugins
            .load("bad.lua", r#"sandy.world { name = "Nothing" }"#)
            .unwrap_err();
        assert!(err.contains("generate"), "{err}");
    }

    #[test]
    fn noise_from_a_script_is_seeded_and_the_grid_is_by_row() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "noise.lua",
                r#"
                local a = sandy.noise { seed = 3, frequency = 0.05, octaves = 2 }
                local b = sandy.noise { seed = 3, frequency = 0.05, octaves = 2 }
                local c = sandy.noise { seed = 4, frequency = 0.05, octaves = 2 }
                assert(a:at(10, 20) == b:at(10, 20), "same seed, same noise")
                assert(a:at(10, 20) ~= c:at(10, 20), "another seed, other noise")
                local g = a:grid(30, 25)
                assert(math.abs(g[20][10] - a:at(10, 20)) < 1e-4, "rows then columns")
                assert(g[24][29] ~= nil and g[25] == nil and g[0][30] == nil)
                "#,
            )
            .unwrap();
        let err = plugins
            .load("noise.lua", r#"sandy.noise { kind = "static" }"#)
            .unwrap_err();
        assert!(err.contains("static"), "{err}");
    }

    #[test]
    fn the_sandbox_reaches_nothing_outside_the_interpreter() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "probe.lua",
                r#"
                assert(io == nil, "io")
                assert(os == nil, "os")
                assert(require == nil, "require")
                assert(debug == nil, "debug")
                assert(math.sqrt(16) == 4, "math is still there")
                assert(sandy.width > 0 and sandy.height > 0, "the grid size is known")
                assert(sandy.find("water") == 3, "built-in materials can be found")
                print("this goes to the log, not stdout")
                "#,
            )
            .unwrap();
    }
}
