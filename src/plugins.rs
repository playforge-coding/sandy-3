//! JavaScript plugins: scripts that add materials, brushes, tools and worlds
//! while the game runs.
//!
//! A plugin is one `.js` file. Dropped on the window, left in the `plugins`
//! folder next to where the game is run from, or compiled in as one of
//! [`BUILTIN`], it is run once as a module with a `sandy` object in scope, and
//! whatever it registers through that object is in the game from then on. A
//! material is a row in the [`Registry`] that
//! [`crate::sim::Simulation::set_tables`] then uploads, a brush or a tool is a
//! function the app calls every frame the mouse is held with it selected, and
//! a world is a function that paints a whole landscape when the panel asks for
//! it.
//!
//! # What a script sees
//!
//! ```js
//! const acid = sandy.material({
//!     name: "Acid", color: [120, 230, 60], density: 120,
//!     mobile: true, liquid: true, spread: 200, glow: true,
//! });
//! sandy.rule({ actor: "Stone", trigger: acid, product: "Empty",
//!              look: "around", chance: 6 });
//! sandy.brush({ name: "Dot", onDrag: (t) => {
//!     sandy.paint(t.x, t.y, 0, t.material);
//! } });
//! sandy.tool({ name: "Fan", onDrag: (t) => {
//!     sandy.wind(t.x, t.y, 30, 0, -4);
//! } });
//! sandy.world({ name: "Flat", generate: (w) => {
//!     const hills = sandy.noise({ seed: w.seed, frequency: 0.01, octaves: 4 });
//!     for (let x = 0; x < w.width; x++) {
//!         const top = w.height * 0.6 - hills.at(x, 0) * 40;
//!         w.fill(x, top, x, w.height - 1, "Soil");
//!     }
//! } });
//! sandy.find("Water")                   // an id, or undefined
//! sandy.paint(x, y, radius, material)   // material is a name or an id
//! sandy.wind(x, y, radius, dvx, dvy)
//! sandy.noise({ seed: 1, frequency: 0.01, octaves: 4 })
//! sandy.width, sandy.height             // the grid, in cells
//! ```
//!
//! Materials, brushes, tools and worlds go by name. Registering a name that
//! already exists replaces the old entry in place, which is what makes
//! dropping a file on the window a second time a reload, and lets a plugin
//! retune a built-in material or brush. Each load is a module of its own, so
//! a `const` at the top of a plugin does not clash with itself on reload.
//!
//! A brush and a tool are the same thing to this module, a function called
//! once a frame while the mouse is held (see [`Kind`]). The difference is what
//! the panel does with them: a brush paints the material picked in the panel,
//! in its own way, so a material and a brush are chosen together; a tool does
//! something else with the cursor. Even the plain brush is a script,
//! `disk.js`, so there is one way to paint rather than a built-in way and a
//! plugin way.
//!
//! # Why a script never touches the world itself
//!
//! Materials are data the kernels read, so a script cannot give one an
//! `update` function any more than a Rust material can (see
//! [`crate::materials`]). What it can do is what the built-in materials do:
//! fill in the properties and the reaction rules. A tool is different. It runs
//! on the CPU once a frame, at cursor rate, so a real JavaScript function is
//! fine there. Even so, the function is not handed the simulation.
//! `sandy.paint` and `sandy.wind` queue a [`Command`], and the app drains the
//! queue and applies it once the script has returned, so the script engine and
//! the GPU state never have to know about each other.
//!
//! A world is the same idea at a larger scale. Its `generate` function paints
//! into a [`Canvas`] in main memory, and when it returns the whole grid goes
//! to the GPU in one write (see [`Plugins::generate`]). The canvas is handed
//! to the script as an object, `w`, that only works while that one generation
//! is running.
//!
//! # Sandbox
//!
//! The engine is QuickJS with the standard JavaScript library and nothing
//! else: no file system, no network, no `require` or `import` of anything
//! outside the script, since none of those exist in the engine unless the
//! host adds them. What the host adds is `sandy`, `console` (whose output
//! goes to the log), `print` as another name for `console.log`, and
//! `assert(condition, message)`. `Math.random` is the game's own generator,
//! reseeded from the world seed before each world is built, so a world that
//! scatters things with it is still the same world for the same seed. A
//! plugin is something the user chose to drop on the window, so this is not
//! a security boundary, but a plugin has no business with any of that and a
//! broken one should not be able to do much harm.
//!
//! # How the engine is driven
//!
//! rquickjs only lets the engine be used inside [`Context::with`], and that
//! call cannot be nested: a Rust function a script calls is already inside
//! one. So every operation here comes in two: a public method that opens the
//! context, and a `*_in` method that takes a [`Ctx`] it was given and does
//! the work. The control script in [`crate::scripting`] runs its requests
//! from inside a script call and uses the `*_in` half.

use std::cell::{Ref, RefCell};
use std::fmt;
use std::path::Path;
use std::rc::Rc;

use rquickjs::class::{Trace, Tracer};
use rquickjs::function::{Opt, Rest};
use rquickjs::{
    CaughtError, Class, Coerced, Context, Ctx, Error, Exception, FromJs, Function, JsLifetime,
    Module, Object, Persistent, Runtime, Type, Value,
};

use crate::materials::{Look, MaterialId, MaterialInfo, Registry, Rule};
use crate::sim::{Grid, Simulation};
use crate::worldgen::{Canvas, Noise};

/// The widest brush a script can ask for, in cells. The paint and gust kernels
/// are dispatched over the brush's bounding square, so an unbounded radius
/// would be an unbounded amount of GPU work for one call.
const MAX_RADIUS: i32 = 512;

/// Where the user's own plugins are looked for at startup, relative to the
/// working directory. Every `.js` file in it is loaded, in name order, after
/// the built-in ones.
pub const PLUGIN_DIR: &str = "plugins";

/// The plugins that ship with the game, compiled into the binary so they are
/// there wherever it is run from. They are ordinary scripts in `src/plugins/`
/// and go through the same loader as a dropped file, which also makes them the
/// worked examples of what a plugin can do.
pub const BUILTIN: &[(&str, &str)] = &[
    ("acid.js", include_str!("plugins/acid.js")),
    ("disk.js", include_str!("plugins/disk.js")),
    ("fan.js", include_str!("plugins/fan.js")),
    // Steam's rules name fire, so fire has to be registered first.
    ("fire.js", include_str!("plugins/fire.js")),
    ("spray.js", include_str!("plugins/spray.js")),
    ("steam.js", include_str!("plugins/steam.js")),
    // Wood's rules name fire too.
    ("wood.js", include_str!("plugins/wood.js")),
    // The worlds only name materials when they are generated, so they could
    // go anywhere, but the panel lists them in this order.
    ("worlds.js", include_str!("plugins/worlds.js")),
    ("caverns.js", include_str!("plugins/caverns.js")),
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

/// A JavaScript function kept outside the engine's context, which is how a
/// handle to one survives between calls.
type Callback = Persistent<Function<'static>>;

/// A brush or a tool a script registered.
struct PluginTool {
    name: String,
    on_drag: Callback,
}

/// A world preset a script registered: a name for the panel and the function
/// that paints it.
struct PluginWorld {
    name: String,
    generate: Callback,
}

/// Everything the `sandy` functions write to. The closures behind them each
/// hold a handle to this, and so does [`Plugins`], which is how what a script
/// did gets back out.
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
    /// What `Math.random` draws from. JavaScript has no way to seed its own,
    /// and a world has to be the same for the same seed.
    rng: fastrand::Rng,
    /// The size of the world a script sees, as `sandy.width` and
    /// `sandy.height`, and the size of the canvas a world is painted on.
    grid: Grid,
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
fn register<'js>(
    ctx: &Ctx<'js>,
    shared: &Rc<RefCell<Shared>>,
    kind: Kind,
    spec: &Object<'js>,
) -> rquickjs::Result<()> {
    let name: String = required(ctx, spec, "name")?;
    if name.trim().is_empty() {
        return Err(fail(ctx, format!("a {} needs a name", kind.word())));
    }
    let on_drag: Function = required(ctx, spec, "onDrag")?;
    let on_drag = Persistent::save(ctx, on_drag);
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
fn register_world<'js>(
    ctx: &Ctx<'js>,
    shared: &Rc<RefCell<Shared>>,
    spec: &Object<'js>,
) -> rquickjs::Result<()> {
    let name: String = required(ctx, spec, "name")?;
    if name.trim().is_empty() {
        return Err(fail(ctx, "a world needs a name"));
    }
    let generate: Function = required(ctx, spec, "generate")?;
    let generate = Persistent::save(ctx, generate);
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
#[derive(JsLifetime)]
#[rquickjs::class]
struct World {
    shared: Rc<RefCell<Shared>>,
    seed: u32,
    /// The generation this handle belongs to; see [`Shared::generation`].
    generation: u64,
}

/// Nothing in a world handle is a JavaScript value, so there is nothing for
/// the garbage collector to follow.
impl<'js> Trace<'js> for World {
    fn trace<'a>(&self, _tracer: Tracer<'a, 'js>) {}
}

impl World {
    /// The canvas, as long as this handle's generation is the one running.
    fn canvas<'a>(
        &self,
        ctx: &Ctx<'_>,
        shared: &'a mut Shared,
    ) -> rquickjs::Result<&'a mut Canvas> {
        if shared.generation != self.generation {
            return Err(fail(ctx, "this world has already been built"));
        }
        shared
            .canvas
            .as_mut()
            .ok_or_else(|| fail(ctx, "this world has already been built"))
    }

    /// Run `f` on the canvas, with a material a script named resolved to its
    /// id first, since both live in the same borrow.
    fn paint(
        &self,
        ctx: &Ctx<'_>,
        material: &Value<'_>,
        f: impl FnOnce(&mut Canvas, MaterialId),
    ) -> rquickjs::Result<()> {
        let shared = &mut *self.shared.borrow_mut();
        let material = resolve(ctx, &shared.registry, material, "material")?;
        let canvas = self.canvas(ctx, shared)?;
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

#[rquickjs::methods]
impl World {
    #[qjs(get)]
    fn seed(&self) -> u32 {
        self.seed
    }

    #[qjs(get)]
    fn width(&self) -> u32 {
        self.shared.borrow().grid.width
    }

    #[qjs(get)]
    fn height(&self) -> u32 {
        self.shared.borrow().grid.height
    }

    /// Put a material in one cell.
    fn set<'js>(
        &self,
        ctx: Ctx<'js>,
        x: f64,
        y: f64,
        material: Value<'js>,
    ) -> rquickjs::Result<()> {
        self.paint(&ctx, &material, |canvas, m| canvas.set(cell(x), cell(y), m))
    }

    /// The material at a cell, or undefined off the grid.
    fn get(&self, ctx: Ctx<'_>, x: f64, y: f64) -> rquickjs::Result<Option<MaterialId>> {
        let shared = &mut *self.shared.borrow_mut();
        Ok(self.canvas(&ctx, shared)?.get(cell(x), cell(y)))
    }

    /// Fill a rectangle, both corners included.
    fn fill<'js>(
        &self,
        ctx: Ctx<'js>,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        material: Value<'js>,
    ) -> rquickjs::Result<()> {
        self.paint(&ctx, &material, |canvas, m| {
            canvas.fill(cell(x0), cell(y0), cell(x1), cell(y1), m)
        })
    }

    /// Fill a circle, the way the plain brush does.
    fn disk<'js>(
        &self,
        ctx: Ctx<'js>,
        x: f64,
        y: f64,
        radius: f64,
        material: Value<'js>,
    ) -> rquickjs::Result<()> {
        let radius = clamp_radius(radius) as i64;
        self.paint(&ctx, &material, |canvas, m| {
            canvas.disk(cell(x), cell(y), radius, m)
        })
    }
}

/// The JavaScript engine, the materials, brushes and tools the scripts have
/// registered, and the commands they have queued.
pub struct Plugins {
    // Declared before the context so it is dropped first: what it holds are
    // handles into the engine, and the engine has to still be there to give
    // them back.
    shared: Rc<RefCell<Shared>>,
    context: Context,
}

impl Default for Plugins {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Plugins {
    fn drop(&mut self) {
        // The closures behind the `sandy` functions hold the shared state,
        // and the shared state holds functions the engine owns. The engine
        // only goes away once nothing points into it, so the cycle is cut
        // here, from this side.
        let mut shared = self.shared.borrow_mut();
        shared.brushes.clear();
        shared.tools.clear();
        shared.worlds.clear();
    }
}

impl Plugins {
    /// An engine with the `sandy` object in it and the built-in materials in
    /// its registry, and no scripts loaded yet. The world it describes to
    /// scripts is the desktop one until [`Plugins::set_grid`] says otherwise.
    pub fn new() -> Self {
        Self::build().expect("build the JavaScript engine")
    }

    fn build() -> rquickjs::Result<Self> {
        let runtime = Runtime::new()?;
        let context = Context::full(&runtime)?;
        let shared = Rc::new(RefCell::new(Shared {
            registry: Registry::builtin(),
            brushes: Vec::new(),
            tools: Vec::new(),
            worlds: Vec::new(),
            commands: Vec::new(),
            canvas: None,
            generation: 0,
            report: Report::default(),
            rng: fastrand::Rng::with_seed(0),
            grid: Grid::DESKTOP,
        }));
        context.with(|ctx| {
            install_globals(&ctx, Sink::Log)?;
            install_sandy(&ctx, &shared)
        })?;
        Ok(Plugins { shared, context })
    }

    /// The size of the world scripts are told about.
    pub fn grid(&self) -> Grid {
        self.shared.borrow().grid
    }

    /// Tell scripts the world is `grid` cells: what `sandy.width` and
    /// `sandy.height` say, and the size of the canvas a world paints. A
    /// phone's grid is only known once its screen is, so this comes after
    /// [`Plugins::new`] and before any script that might read them.
    pub fn set_grid(&mut self, grid: Grid) {
        self.shared.borrow_mut().grid = grid;
        self.context
            .with(|ctx| -> rquickjs::Result<()> {
                let sandy: Object = ctx.globals().get("sandy")?;
                sandy.set("width", grid.width)?;
                sandy.set("height", grid.height)
            })
            .expect("the sandy object is always there");
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

    /// Load every `.js` file in [`PLUGIN_DIR`], in name order, and say how
    /// each went: its file name, and the report or the error. No folder is
    /// simply no plugins; a script that fails does not stop the rest loading.
    pub fn load_dir(&mut self) -> Vec<(String, Result<Report, String>)> {
        let Ok(entries) = std::fs::read_dir(PLUGIN_DIR) else {
            return Vec::new();
        };
        let mut paths: Vec<_> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "js"))
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
        self.context.with(|ctx| self.load_file_in(&ctx, path))
    }

    /// [`Plugins::load_file`], from inside the engine.
    pub(crate) fn load_file_in(&self, ctx: &Ctx<'_>, path: &Path) -> Result<Report, String> {
        let source =
            std::fs::read_to_string(path).map_err(|err| format!("could not read it: {err}"))?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.load_in(ctx, &name, &source)
    }

    /// Run `source` as a script. `name` is what the engine puts in front of
    /// the line number in an error message.
    pub fn load(&mut self, name: &str, source: &str) -> Result<Report, String> {
        self.context.with(|ctx| self.load_in(&ctx, name, source))
    }

    /// [`Plugins::load`], from inside the engine.
    pub(crate) fn load_in(
        &self,
        ctx: &Ctx<'_>,
        name: &str,
        source: &str,
    ) -> Result<Report, String> {
        self.shared.borrow_mut().report = Report::default();
        let result = (|| -> rquickjs::Result<()> {
            let module = Module::declare(ctx.clone(), name, source)?;
            let (_, finished) = module.eval()?;
            // A plugin runs to the end as it is loaded. Awaiting something
            // that resolves on its own is fine; awaiting something that
            // never comes is the one way to get here with the promise still
            // pending.
            finished.finish::<()>().map_err(|err| match err {
                Error::WouldBlock => fail(
                    ctx,
                    "the plugin is waiting on something that never comes; a plugin has to finish as it is loaded",
                ),
                other => other,
            })
        })();
        let report = self.shared.borrow().report;
        result.map(|()| report).map_err(|err| describe(ctx, err))
    }

    /// The engine's context, for the control script in [`crate::scripting`],
    /// which runs in it alongside the plugins.
    pub(crate) fn context(&self) -> &Context {
        &self.context
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
        self.context
            .with(|ctx| self.run_in(&ctx, kind, index, stroke))
    }

    /// [`Plugins::run`], from inside the engine.
    pub(crate) fn run_in(
        &self,
        ctx: &Ctx<'_>,
        kind: Kind,
        index: usize,
        stroke: Stroke,
    ) -> Result<(), String> {
        // The function is cloned out so the borrow is released before the
        // call, since the script will want to borrow the queue itself.
        let on_drag = self
            .shared
            .borrow()
            .list(kind)
            .get(index)
            .map(|entry| entry.on_drag.clone())
            .ok_or_else(|| format!("there is no {} number {index}", kind.word()))?;
        (|| -> rquickjs::Result<()> {
            let on_drag = on_drag.restore(ctx)?;
            let t = Object::new(ctx.clone())?;
            t.set("x", stroke.x)?;
            t.set("y", stroke.y)?;
            t.set("px", stroke.px)?;
            t.set("py", stroke.py)?;
            t.set("first", stroke.first)?;
            t.set("radius", stroke.radius)?;
            t.set("material", stroke.material)?;
            on_drag.call::<_, ()>((t,))
        })()
        .map_err(|err| describe(ctx, err))
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
    /// The same world and seed always give the same grid. `Math.random` is
    /// reseeded from `seed` first, so a script can scatter trees with it
    /// and still be reproducible; the noise it makes carries its own seed.
    pub fn generate(&mut self, index: usize, seed: u32) -> Result<Vec<MaterialId>, String> {
        self.context.with(|ctx| self.generate_in(&ctx, index, seed))
    }

    /// [`Plugins::generate`], from inside the engine.
    pub(crate) fn generate_in(
        &self,
        ctx: &Ctx<'_>,
        index: usize,
        seed: u32,
    ) -> Result<Vec<MaterialId>, String> {
        let generate = self
            .shared
            .borrow()
            .worlds
            .get(index)
            .map(|entry| entry.generate.clone())
            .ok_or_else(|| format!("there is no world number {index}"))?;
        let generation = {
            let mut shared = self.shared.borrow_mut();
            let grid = shared.grid;
            shared.canvas = Some(Canvas::new(grid.width, grid.height));
            shared.generation += 1;
            shared.rng.seed(u64::from(seed));
            shared.generation
        };

        let result = (|| -> rquickjs::Result<()> {
            let generate = generate.restore(ctx)?;
            let world = Class::instance(
                ctx.clone(),
                World {
                    shared: self.shared.clone(),
                    seed,
                    generation,
                },
            )?;
            generate.call::<_, ()>((world,))
        })();
        // The canvas comes out whatever happened, so a script that failed
        // halfway leaves nothing behind for the next one to paint over.
        let canvas = self.shared.borrow_mut().canvas.take();
        result.map_err(|err| describe(ctx, err))?;
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

/// Put the `sandy` object in the globals, with every function on it holding
/// a handle to `shared`.
fn install_sandy<'js>(ctx: &Ctx<'js>, shared: &Rc<RefCell<Shared>>) -> rquickjs::Result<()> {
    let sandy = Object::new(ctx.clone())?;
    let grid = shared.borrow().grid;
    sandy.set("width", grid.width)?;
    sandy.set("height", grid.height)?;

    let s = shared.clone();
    sandy.set(
        "material",
        Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, spec: Object<'js>| -> rquickjs::Result<MaterialId> {
                let info = material_from(&ctx, &spec)?;
                let mut shared = s.borrow_mut();
                let id = shared
                    .registry
                    .add_material(info)
                    .map_err(|message| fail(&ctx, message))?;
                shared.report.materials += 1;
                Ok(id)
            },
        )?,
    )?;

    let s = shared.clone();
    sandy.set(
        "rule",
        Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, spec: Object<'js>| -> rquickjs::Result<()> {
                let shared = &mut *s.borrow_mut();
                let actor = resolve(&ctx, &shared.registry, &spec.get("actor")?, "actor")?;
                let trigger = resolve(&ctx, &shared.registry, &spec.get("trigger")?, "trigger")?;
                let product = resolve(&ctx, &shared.registry, &spec.get("product")?, "product")?;
                let look = look_from(&ctx, optional(&ctx, &spec, "look", None)?)?;
                let chance: u32 = optional(&ctx, &spec, "chance", 1)?;
                if chance == 0 {
                    return Err(fail(
                        &ctx,
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
            },
        )?,
    )?;

    for (key, kind) in [("brush", Kind::Brush), ("tool", Kind::Tool)] {
        let s = shared.clone();
        sandy.set(
            key,
            Function::new(ctx.clone(), move |ctx: Ctx<'js>, spec: Object<'js>| {
                register(&ctx, &s, kind, &spec)
            })?,
        )?;
    }

    let s = shared.clone();
    sandy.set(
        "world",
        Function::new(ctx.clone(), move |ctx: Ctx<'js>, spec: Object<'js>| {
            register_world(&ctx, &s, &spec)
        })?,
    )?;

    sandy.set(
        "noise",
        Function::new(
            ctx.clone(),
            |ctx: Ctx<'js>, spec: Object<'js>| -> rquickjs::Result<Class<'js, Noise>> {
                let noise = Noise::from_spec(&ctx, &spec)?;
                Class::instance(ctx, noise)
            },
        )?,
    )?;

    let s = shared.clone();
    sandy.set(
        "find",
        Function::new(ctx.clone(), move |name: String| -> Option<MaterialId> {
            s.borrow().registry.find(&name)
        })?,
    )?;

    let s = shared.clone();
    sandy.set(
        "paint",
        Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, x: f64, y: f64, radius: f64, material: Value<'js>| {
                let mut shared = s.borrow_mut();
                let material = resolve(&ctx, &shared.registry, &material, "material")?;
                shared.commands.push(Command::Paint {
                    x: x.round() as i32,
                    y: y.round() as i32,
                    radius: clamp_radius(radius),
                    material,
                });
                Ok::<(), Error>(())
            },
        )?,
    )?;

    let s = shared.clone();
    sandy.set(
        "wind",
        Function::new(
            ctx.clone(),
            move |x: f64, y: f64, radius: f64, dvx: f64, dvy: f64| {
                s.borrow_mut().commands.push(Command::Wind {
                    x: x.round() as i32,
                    y: y.round() as i32,
                    radius: clamp_radius(radius),
                    dvx: dvx as f32,
                    dvy: dvy as f32,
                });
            },
        )?,
    )?;

    ctx.globals().set("sandy", sandy)?;

    // `Math.random` is replaced with the game's own generator, which is what
    // lets a world be reseeded (see `Plugins::generate`).
    let s = shared.clone();
    let math: Object = ctx.globals().get("Math")?;
    math.set(
        "random",
        Function::new(ctx.clone(), move || -> f64 { s.borrow_mut().rng.f64() })?,
    )?;
    Ok(())
}

/// Where `console` writes.
#[derive(Clone, Copy)]
pub(crate) enum Sink {
    /// The log, which is where a plugin's output belongs: nobody is watching
    /// stdout while the window is up.
    Log,
    /// Standard output, for a control script run from a terminal, whose
    /// output is the point.
    Stdout,
}

/// Put `console`, `print` and `assert` in the globals. QuickJS has none of
/// them of its own; `console` is a host's to provide.
pub(crate) fn install_globals<'js>(ctx: &Ctx<'js>, sink: Sink) -> rquickjs::Result<()> {
    let console = Object::new(ctx.clone())?;
    for (name, level) in [
        ("log", log::Level::Info),
        ("info", log::Level::Info),
        ("debug", log::Level::Debug),
        ("warn", log::Level::Warn),
        ("error", log::Level::Error),
    ] {
        let function = Function::new(ctx.clone(), move |ctx: Ctx<'js>, args: Rest<Value<'js>>| {
            let text = args
                .0
                .iter()
                .map(|value| show(&ctx, value))
                .collect::<Vec<_>>()
                .join(" ");
            match sink {
                Sink::Log => log::log!(target: "plugin", level, "{text}"),
                // Warnings and errors go to stderr, as they do in a
                // browser or Node. `log::Level` counts errors lowest.
                Sink::Stdout if level <= log::Level::Warn => eprintln!("{text}"),
                Sink::Stdout => {
                    use std::io::Write as _;
                    let mut out = std::io::stdout().lock();
                    let _ = writeln!(out, "{text}");
                    let _ = out.flush();
                }
            }
        })?;
        if name == "log" {
            ctx.globals().set("print", function.clone())?;
        }
        console.set(name, function)?;
    }
    ctx.globals().set("console", console)?;

    ctx.globals().set(
        "assert",
        Function::new(
            ctx.clone(),
            |ctx: Ctx<'js>, condition: Coerced<bool>, message: Opt<Value<'js>>| {
                if condition.0 {
                    Ok(())
                } else {
                    let message = message
                        .0
                        .map(|value| show(&ctx, &value))
                        .unwrap_or_else(|| "assertion failed".to_string());
                    Err(fail(&ctx, message))
                }
            },
        )?,
    )?;
    Ok(())
}

/// A value as `console.log` shows it: a string as it is, an object as JSON,
/// and anything else the way JavaScript would turn it into a string.
fn show<'js>(ctx: &Ctx<'js>, value: &Value<'js>) -> String {
    if let Some(string) = value.as_string() {
        return string.to_string().unwrap_or_default();
    }
    if value.is_object()
        && !value.is_function()
        && !value.is_error()
        && let Ok(Some(json)) = ctx.json_stringify(value.clone())
        && let Ok(text) = json.to_string()
    {
        return text;
    }
    Coerced::<String>::from_js(ctx, value.clone())
        .map(|text| text.0)
        .unwrap_or_else(|_| value.type_name().to_string())
}

/// An error for the script, raised where it made the call.
pub(crate) fn fail(ctx: &Ctx<'_>, message: impl fmt::Display) -> Error {
    Exception::throw_message(ctx, &message.to_string())
}

/// A field of a spec object, or nothing if it is missing, null or undefined.
/// A field of the wrong type is an error naming the field.
pub(crate) fn field<'js, T: FromJs<'js>>(
    ctx: &Ctx<'js>,
    spec: &Object<'js>,
    key: &str,
) -> rquickjs::Result<Option<T>> {
    let value: Value<'js> = spec.get(key)?;
    if value.type_of().is_void() {
        return Ok(None);
    }
    let found = value.type_name();
    T::from_js(ctx, value).map(Some).map_err(|err| match err {
        Error::Exception => Error::Exception,
        Error::FromJs { to, message, .. } => {
            let mut text = format!("'{key}' should be {}, not a {found}", kind_of(to));
            if let Some(message) = message.filter(|message| !message.is_empty()) {
                text.push_str(&format!(" ({message})"));
            }
            fail(ctx, text)
        }
        other => fail(ctx, format!("'{key}': {other}")),
    })
}

/// The Rust type a conversion wanted, said the way a script would put it.
fn kind_of(to: &str) -> &str {
    match to {
        "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "f32" | "f64" | "usize"
        | "isize" | "int" | "float" | "number" => "a number",
        "bool" => "a boolean",
        _ if to.eq_ignore_ascii_case("string") => "a string",
        _ if to.eq_ignore_ascii_case("function") => "a function",
        _ if to.eq_ignore_ascii_case("array") || to.starts_with("Vec") => "an array",
        _ if to.eq_ignore_ascii_case("object") => "an object",
        other => other,
    }
}

/// A field that has to be there.
fn required<'js, T: FromJs<'js>>(
    ctx: &Ctx<'js>,
    spec: &Object<'js>,
    key: &str,
) -> rquickjs::Result<T> {
    field(ctx, spec, key)?.ok_or_else(|| fail(ctx, format!("'{key}' is required")))
}

/// A field that may be missing, in which case `default` stands in.
pub(crate) fn optional<'js, T: FromJs<'js>>(
    ctx: &Ctx<'js>,
    spec: &Object<'js>,
    key: &str,
    default: T,
) -> rquickjs::Result<T> {
    Ok(field(ctx, spec, key)?.unwrap_or(default))
}

fn material_from<'js>(ctx: &Ctx<'js>, spec: &Object<'js>) -> rquickjs::Result<MaterialInfo> {
    let name: String = required(ctx, spec, "name")?;
    if name.trim().is_empty() {
        return Err(fail(ctx, "a material needs a name"));
    }
    let color: Vec<f64> = required(ctx, spec, "color")?;
    let channel = |v: f64| (v.is_finite() && (0.0..=255.0).contains(&v)).then_some(v as u8);
    let [Some(r), Some(g), Some(b)] = color.iter().map(|&v| channel(v)).collect::<Vec<_>>()[..]
    else {
        return Err(fail(
            ctx,
            "'color' should be three numbers from 0 to 255, like [200, 120, 40]",
        ));
    };
    Ok(MaterialInfo {
        // Material names are `&'static str` because the built-in ones are
        // constants. A script's material lives as long as the game does too,
        // so its name is simply never freed. Reloading a script leaks the few
        // bytes of the old name, which is nothing.
        name: name.leak(),
        color: [r, g, b],
        jitter: optional(ctx, spec, "jitter", 0)?,
        density: required(ctx, spec, "density")?,
        mobile: optional(ctx, spec, "mobile", false)?,
        passable: optional(ctx, spec, "passable", true)?,
        liquid: optional(ctx, spec, "liquid", false)?,
        spread: optional(ctx, spec, "spread", 0)?,
        windborne: optional(ctx, spec, "windborne", false)?,
        glow: optional(ctx, spec, "glow", false)?,
        draft: optional(ctx, spec, "draft", 0)?,
    })
}

/// A material named by a script: its id, or its name in any case.
pub(crate) fn resolve(
    ctx: &Ctx<'_>,
    registry: &Registry,
    value: &Value<'_>,
    what: &str,
) -> rquickjs::Result<MaterialId> {
    let by_id = |id: f64| {
        if id.fract() == 0.0 && (0.0..registry.materials().len() as f64).contains(&id) {
            Ok(id as MaterialId)
        } else {
            Err(fail(
                ctx,
                format!("{what}: there is no material with id {id}"),
            ))
        }
    };
    match value.type_of() {
        Type::Int | Type::Float => by_id(value.as_number().unwrap_or(f64::NAN)),
        Type::String => {
            let name = value.as_string().expect("a string value").to_string()?;
            registry
                .find(&name)
                .ok_or_else(|| fail(ctx, format!("{what}: there is no material called '{name}'")))
        }
        Type::Undefined | Type::Null | Type::Uninitialized => {
            Err(fail(ctx, format!("'{what}' is required")))
        }
        other => Err(fail(
            ctx,
            format!("{what} should be a material name or id, not a {other}"),
        )),
    }
}

fn look_from(ctx: &Ctx<'_>, name: Option<String>) -> rquickjs::Result<Look> {
    match name.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("ortho") => Ok(Look::Ortho),
        Some("around") => Ok(Look::Around),
        Some("above") => Ok(Look::Above),
        Some("below") => Ok(Look::Below),
        Some(other) => Err(fail(
            ctx,
            format!("look should be ortho, around, above or below, not '{other}'"),
        )),
    }
}

pub(crate) fn clamp_radius(radius: f64) -> i32 {
    if radius.is_finite() {
        (radius.round() as i32).clamp(0, MAX_RADIUS)
    } else {
        0
    }
}

/// The one line of an error worth showing on the panel: what went wrong, and
/// where in the script if that can be told. The engine keeps a stack with
/// every error it throws, with the file and line of each frame, and the
/// topmost frame that is in a script rather than in the host is the line
/// worth pointing at. This takes the error out of the engine, so it has to
/// be called from where the error was made.
pub(crate) fn describe(ctx: &Ctx<'_>, err: Error) -> String {
    match CaughtError::from_error(ctx, err) {
        CaughtError::Exception(exception) => {
            let name = exception
                .get::<_, Option<Coerced<String>>>("name")
                .ok()
                .flatten()
                .map(|name| name.0)
                .unwrap_or_default();
            let message = exception.message().unwrap_or_default();
            let mut text = match (name.as_str(), message.as_str()) {
                ("" | "Error", message) => message.to_string(),
                (name, "") => name.to_string(),
                (name, message) => format!("{name}: {message}"),
            };
            if text.is_empty() {
                text = "an error with no message".to_string();
            }
            if let Some(place) = exception.stack().as_deref().and_then(place) {
                text = format!("{text} (at {place})");
            }
            text
        }
        // `throw "a string"` is legal, and all there is to say is the string.
        CaughtError::Value(value) => show(ctx, &value),
        CaughtError::Error(err) => err.to_string(),
    }
}

/// The first frame of a stack that names a place in a script, as
/// `file:line:column`. The engine writes a frame as `at name (file:line:col)`
/// or, for a syntax error, `at file:line:col`, and a frame in the host as
/// `at name (native)`, which is skipped.
fn place(stack: &str) -> Option<String> {
    stack.lines().find_map(|line| {
        let line = line.trim().strip_prefix("at ")?;
        let inner = match line.rsplit_once(" (") {
            Some((_, rest)) => rest.strip_suffix(')').unwrap_or(rest),
            None => line,
        };
        let (head, last) = inner.rsplit_once(':')?;
        last.parse::<u32>().ok()?;
        (!head.is_empty()).then(|| inner.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{EMPTY, LAVA, SAND, SOIL, STONE, WATER};
    use crate::sim::{GRID_H, GRID_W};

    #[test]
    fn scripts_are_told_the_size_of_the_world_and_paint_to_it() {
        // Until told otherwise a script sees the desktop grid. A phone tells
        // the plugins its own grid before they load, and from then on that is
        // what `sandy.width` says and the size of the canvas a world paints.
        let mut plugins = Plugins::new();
        assert_eq!(plugins.grid(), Grid::DESKTOP);
        plugins
            .load(
                "size.js",
                "assert(sandy.width === 3000 && sandy.height === 1500);",
            )
            .unwrap();

        let phone = Grid {
            width: 520,
            height: 1156,
        };
        plugins.set_grid(phone);
        assert_eq!(plugins.grid(), phone);
        plugins
            .load(
                "flat.js",
                r#"
                assert(sandy.width === 520 && sandy.height === 1156);
                sandy.world({ name: "Flat", generate: (w) => {
                    assert(w.width === 520 && w.height === 1156);
                    w.fill(0, w.height - 1, w.width - 1, w.height - 1, "Stone");
                } });
                "#,
            )
            .unwrap();
        let cells = plugins.generate(0, 1).unwrap();
        assert_eq!(cells.len(), phone.cells());
        assert_eq!(cells.iter().filter(|&&m| m == STONE).count(), 520);
    }

    #[test]
    fn a_script_can_add_a_material_and_a_rule() {
        let mut plugins = Plugins::new();
        let report = plugins
            .load(
                "acid.js",
                r#"
                const acid = sandy.material({
                    name: "Acid", color: [1, 2, 3], density: 120,
                    mobile: true, liquid: true, spread: 200, glow: true,
                });
                sandy.rule({ actor: "stone", trigger: acid, product: "Empty",
                             look: "around", chance: 6 });
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
        // A `const` at the top level, which would be a redeclaration if the
        // second load ran in the same scope as the first.
        let script = |color: &str| {
            r#"const ash = sandy.material({ name: "Ash", color: COLOR, density: 100, mobile: true });
               sandy.tool({ name: "Puff", onDrag: (t) => {} });"#
                .replace("COLOR", color)
        };
        plugins.load("ash.js", &script("[9, 9, 9]")).unwrap();
        let count = plugins.registry().materials().len();
        let id = plugins.registry().find("Ash").unwrap();

        plugins.load("ash.js", &script("[4, 4, 4]")).unwrap();
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
                "fan.js",
                r#"
                sandy.tool({ name: "Fan", onDrag: (t) => {
                    sandy.wind(t.x, t.y, t.radius * 3, 0, -4);
                    if (t.first) sandy.paint(t.x, t.y, 2.4, t.material);
                    if (t.x !== t.px) sandy.paint(t.px, t.py, 0, "water");
                } });
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
    fn a_broken_script_reports_what_went_wrong_and_the_engine_carries_on() {
        let mut plugins = Plugins::new();

        let err = plugins.load("bad.js", "sandy.material({").unwrap_err();
        assert!(err.contains("bad.js"), "a syntax error says where: {err}");

        let err = plugins
            .load(
                "bad.js",
                r#"sandy.material({ name: "Ash", color: [1, 2, 3] })"#,
            )
            .unwrap_err();
        assert!(err.contains("'density' is required"), "{err}");
        assert!(
            err.contains("bad.js:1"),
            "an error from the host says where: {err}"
        );

        let err = plugins
            .load(
                "bad.js",
                r#"sandy.rule({ actor: "Sand", trigger: "Unobtainium", product: 0 })"#,
            )
            .unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");

        let err = plugins
            .load(
                "bad.js",
                r#"sandy.rule({ actor: "Sand", trigger: "Water", product: 0, look: "sideways" })"#,
            )
            .unwrap_err();
        assert!(err.contains("sideways"), "{err}");

        let err = plugins
            .load(
                "bad.js",
                r#"sandy.material({ name: "Ash", color: [1, 2], density: 5 })"#,
            )
            .unwrap_err();
        assert!(err.contains("color"), "{err}");

        let err = plugins
            .load(
                "bad.js",
                r#"sandy.material({ name: "Ash", color: [1, 2, 3], density: "lots" })"#,
            )
            .unwrap_err();
        assert!(err.contains("'density' should be a number"), "{err}");

        let err = plugins
            .load("bad.js", "throw new Error('no thanks')")
            .unwrap_err();
        assert!(err.contains("no thanks"), "{err}");
        assert!(err.contains("bad.js:1"), "{err}");

        let err = plugins
            .load("bad.js", "\nassert(1 === 2, 'not so')")
            .unwrap_err();
        assert!(err.contains("not so") && err.contains("bad.js:2"), "{err}");

        let err = plugins
            .load("bad.js", "await new Promise(() => {})")
            .unwrap_err();
        assert!(err.contains("never comes"), "{err}");

        plugins
            .load(
                "tool.js",
                "sandy.tool({ name: 'Oops', onDrag: (t) => t.nothing.here })",
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
        assert!(err.contains("tool.js:1"), "{err}");
        assert!(plugins.run(Kind::Tool, 1, stroke).is_err(), "no such tool");
        assert!(
            plugins.run(Kind::Brush, 0, stroke).is_err(),
            "no brushes at all"
        );

        // None of that has hurt the engine or the registry.
        let registry_size = plugins.registry().materials().len();
        assert_eq!(plugins.registry().find("Ash"), None);
        plugins
            .load(
                "good.js",
                r#"sandy.material({ name: "Ash", color: [1, 2, 3], density: 5 })"#,
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
        let acid = registry.find("Acid").expect("acid.js adds Acid");
        assert!(registry.materials()[acid as usize].liquid);
        assert!(
            registry.rules().iter().any(|rule| rule.trigger == acid),
            "acid.js adds rules that fire next to acid"
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
                "flat.js",
                r#"
                sandy.world({ name: "Flat", generate: (w) => {
                    assert(w.width === sandy.width && w.height === sandy.height);
                    assert(w.get(0, 0) === 0, "a fresh canvas is air");
                    assert(w.get(-1, 0) === undefined, "off the grid is undefined");
                    // A floor, a pond in it, a boulder, and one grain of sand
                    // off the edge that goes nowhere.
                    w.fill(0, w.height - 10, w.width - 1, w.height - 1, "Stone");
                    w.fill(100, w.height - 10, 199, w.height - 6, "water");
                    w.disk(500, 100, 3, sandy.find("Soil"));
                    w.set(w.width, 5, "Sand");
                    assert(w.get(500, 100) === sandy.find("Soil"));
                } });
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
                "flat.js",
                r#"sandy.world({ name: "flat", generate: (w) => {} })"#,
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
                "bad.js",
                r#"globalThis.stash = null;
sandy.world({ name: "Half", generate: (w) => {
    globalThis.stash = w;
    w.fill(0, 0, 10, 10, "Stone");
    w.set(0, 0, "Unobtainium");
} });
sandy.world({ name: "Late", generate: (w) => stash.get(0, 0) });
"#,
            )
            .unwrap();
        let err = plugins.generate(0, 1).unwrap_err();
        assert!(err.contains("Unobtainium"), "{err}");
        assert!(err.contains("bad.js:5"), "{err}");

        // The canvas the failed script painted on is gone, and so a handle to
        // it kept from that run is no use to a later one.
        let err = plugins.generate(1, 1).unwrap_err();
        assert!(err.contains("already been built"), "{err}");

        // A world with no generate function is refused at registration.
        let err = plugins
            .load("bad.js", r#"sandy.world({ name: "Nothing" })"#)
            .unwrap_err();
        assert!(err.contains("generate"), "{err}");
    }

    #[test]
    fn noise_from_a_script_is_seeded_and_the_grid_is_by_row() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "noise.js",
                r#"
                const a = sandy.noise({ seed: 3, frequency: 0.05, octaves: 2 });
                const b = sandy.noise({ seed: 3, frequency: 0.05, octaves: 2 });
                const c = sandy.noise({ seed: 4, frequency: 0.05, octaves: 2 });
                assert(a.at(10, 20) === b.at(10, 20), "same seed, same noise");
                assert(a.at(10, 20) !== c.at(10, 20), "another seed, other noise");
                const g = a.grid(30, 25);
                assert(Math.abs(g[20][10] - a.at(10, 20)) < 1e-4, "rows then columns");
                assert(g[24][29] !== undefined && g[25] === undefined && g[0][30] === undefined);
                "#,
            )
            .unwrap();
        let err = plugins
            .load("noise.js", r#"sandy.noise({ kind: "static" })"#)
            .unwrap_err();
        assert!(err.contains("static"), "{err}");
    }

    #[test]
    fn math_random_is_the_games_own_and_follows_the_world_seed() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "dice.js",
                r#"
                sandy.world({ name: "Dice", generate: (w) => {
                    globalThis.rolls = [Math.random(), Math.random()];
                } });
                "#,
            )
            .unwrap();
        let rolls = |plugins: &mut Plugins, seed: u32| -> Vec<f64> {
            plugins.generate(0, seed).unwrap();
            plugins
                .context()
                .with(|ctx| ctx.globals().get::<_, Vec<f64>>("rolls").unwrap())
        };
        let a = rolls(&mut plugins, 7);
        let b = rolls(&mut plugins, 7);
        let c = rolls(&mut plugins, 8);
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
        assert_eq!(a, b, "the same seed rolls the same");
        assert_ne!(a, c);
    }

    #[test]
    fn the_sandbox_reaches_nothing_outside_the_engine() {
        let mut plugins = Plugins::new();
        plugins
            .load(
                "probe.js",
                r#"
                assert(typeof require === "undefined", "require");
                assert(typeof process === "undefined", "process");
                assert(typeof fetch === "undefined", "fetch");
                assert(typeof os === "undefined" && typeof std === "undefined", "quickjs-libc");
                assert(Math.sqrt(16) === 4, "Math is still there");
                assert(sandy.width > 0 && sandy.height > 0, "the grid size is known");
                assert(sandy.find("water") === 3, "built-in materials can be found");
                console.log("this goes to the log, not stdout", { and: "objects" }, [1, 2]);
                print("so does this");
                "#,
            )
            .unwrap();
    }
}
