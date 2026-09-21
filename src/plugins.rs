//! Lua plugins: scripts that add materials, brushes and tools while the game
//! runs.
//!
//! A plugin is one `.lua` file. Dropped on the window, left in the `plugins`
//! folder next to where the game is run from, or compiled in as one of
//! [`BUILTIN`], it is executed once with a `sandy` table in scope, and whatever
//! it registers through that table is in the game from then on. A material is
//! a row in the [`Registry`] that [`crate::sim::Simulation::set_tables`] then
//! uploads, and a brush or a tool is a Lua function the app calls every frame
//! the mouse is held with it selected.
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
//! sandy.find("Water")                   -- an id, or nil
//! sandy.paint(x, y, radius, material)   -- material is a name or an id
//! sandy.wind(x, y, radius, dvx, dvy)
//! sandy.width, sandy.height             -- the grid, in cells
//! ```
//!
//! Materials, brushes and tools go by name. Registering a name that already
//! exists replaces the old entry in place, which is what makes dropping a file
//! on the window a second time a reload, and lets a plugin retune a built-in
//! material or brush.
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

use mlua::{FromLua, Function, Lua, LuaOptions, StdLib, Table, Value, Variadic};

use crate::materials::{Look, MaterialId, MaterialInfo, Registry, Rule};
use crate::sim::{GRID_H, GRID_W};

/// The widest brush a script can ask for, in cells. The paint and gust kernels
/// are dispatched over the brush's bounding square, so an unbounded radius
/// would be an unbounded amount of GPU work for one call.
const MAX_RADIUS: i32 = 512;

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
            "{}, {}, {}, {}",
            count(self.materials, "material", "materials"),
            count(self.rules, "rule", "rules"),
            count(self.brushes, "brush", "brushes"),
            count(self.tools, "tool", "tools")
        )
    }
}

/// A brush or a tool a script registered.
struct PluginTool {
    name: String,
    on_drag: Function,
}

/// Everything the `sandy` functions write to. The Lua closures each hold a
/// handle to this, and so does [`Plugins`], which is how what a script did
/// gets back out.
struct Shared {
    registry: Registry,
    brushes: Vec<PluginTool>,
    tools: Vec<PluginTool>,
    commands: Vec<Command>,
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
            commands: Vec::new(),
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

fn runtime(message: impl fmt::Display) -> mlua::Error {
    mlua::Error::runtime(message.to_string())
}

/// A field that has to be there.
fn required<T: FromLua>(spec: &Table, key: &str) -> mlua::Result<T> {
    optional(spec, key, None)?.ok_or_else(|| runtime(format!("'{key}' is required")))
}

/// A field that may be missing, in which case `default` stands in.
fn optional<T: FromLua>(spec: &Table, key: &str, default: T) -> mlua::Result<T> {
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
fn resolve(registry: &Registry, value: &Value, what: &str) -> mlua::Result<MaterialId> {
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

fn clamp_radius(radius: f64) -> i32 {
    (radius.round() as i32).clamp(0, MAX_RADIUS)
}

/// The one line of an error worth showing on the panel: what went wrong, and
/// where in the script if that can be told. Lua's own errors say where on
/// their first line; an error raised from the Rust side does not, but the
/// traceback under it does, so that is looked for and added.
fn describe(err: &mlua::Error) -> String {
    let text = err.to_string();
    let first = plain(err);
    let place = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("[C]"))
        .find_map(|line| line.split_once(": in ").map(|(place, _)| place));
    match place {
        Some(place) if !first.starts_with(place) => format!("{first} (at {place})"),
        _ => first,
    }
}

/// An error's message with mlua's own framing taken off: the root cause of a
/// callback error, and only the first line of it.
fn plain(err: &mlua::Error) -> String {
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
    use crate::materials::{EMPTY, SAND, STONE, WATER};

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
                tools: 0
            }
        );
        assert_eq!(report.to_string(), "1 material, 1 rule, 0 brushes, 0 tools");

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
