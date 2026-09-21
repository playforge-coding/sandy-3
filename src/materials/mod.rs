//! The material system.
//!
//! Each material lives in **its own file** (`sand.rs`, `stone.rs`, …) and
//! contributes two things: a [`MaterialInfo`] describing what it is, and
//! (optionally) a few [`Rule`]s describing what it turns into when it touches
//! something else. [`table`] collects them, in id order.
//!
//! # Where the behaviour went
//!
//! In a CPU falling-sand engine a material is a trait object with an `update`
//! method. That does not survive the trip to a GPU: the whole grid is stepped by
//! one compute kernel, and a kernel cannot call back into Rust. So a material
//! here is *data*, and the kernels in [`crate::kernels`] are the one piece of
//! logic that reads it. [`MaterialInfo::density`] and the flags decide how a cell
//! moves; the [`Rule`] table decides what it becomes. Neither kernel mentions
//! sand or water by name.
//!
//! Adding a material is therefore still a new file and one line in [`table`] —
//! with no kernel change at all, as long as the existing properties can describe
//! it.
//!
//! The same goes for a material a plugin adds while the game is running: it is
//! one more row in a [`Registry`], which starts out as the built-in tables and
//! is what the host uploads to the GPU. See [`crate::plugins`].
//!
//! # Adding a new material
//!
//! 1. Create `src/materials/<name>.rs` (copy `sand.rs` or `stone.rs`).
//! 2. `mod <name>;` it below.
//! 3. Add it to [`table`].

mod empty;
mod lava;
mod sand;
mod soil;
mod stone;
mod water;

/// A material identifier. `0` is always [`EMPTY`]; every other value indexes
/// into the table built by [`table`].
pub type MaterialId = u8;

/// How many materials there can be, built in and from plugins together. A cell
/// keeps its material in eight bits, so this is as many as an id can name.
pub const MAX_MATERIALS: usize = 256;

/// The empty cell (air / nothing). Always id `0`.
pub const EMPTY: MaterialId = 0;
/// Named ids for the materials that other materials react with, and that the
/// keyboard shortcuts in [`crate::app`] select. These must match the positions
/// in [`table`].
pub const SAND: MaterialId = 1;
pub const STONE: MaterialId = 2;
pub const WATER: MaterialId = 3;
pub const LAVA: MaterialId = 4;
pub const SOIL: MaterialId = 5;

/// A material's static properties: everything the movement kernel needs to push
/// a cell around, and everything the renderer needs to colour it.
#[derive(Clone, Copy)]
pub struct MaterialInfo {
    /// Human-readable name, shown in the material picker.
    pub name: &'static str,
    /// Base colour, RGB, 0–255.
    pub color: [u8; 3],
    /// Per-cell brightness jitter (0 = flat). Gives powders a grainy look.
    pub jitter: u8,
    /// Where this material sits in the up/down ordering. A `mobile` cell sinks
    /// past any `passable` cell that is lighter than it, which is the single
    /// rule behind sand falling through air, sand sinking through water, and
    /// water floating on lava. Air is [`AIR_DENSITY`]; anything below that rises
    /// instead of falling.
    pub density: u8,
    /// Whether this material moves under its own weight. False for air and for
    /// solids, which is what keeps a hillside of soil standing.
    pub mobile: bool,
    /// Whether another material can displace this one. False for solids; true
    /// for air and for everything that flows.
    pub passable: bool,
    /// Whether this material flows sideways to find its own level. Powders pile
    /// up; liquids do not.
    pub liquid: bool,
    /// How readily a liquid creeps sideways, 0–255 — its runniness. Water is
    /// near the top of the range and levels off almost at once; lava is near the
    /// bottom and pools into blobs. Ignored unless `liquid`.
    pub spread: u8,
    /// Whether a gust can shove this material about. Only loose, light things
    /// ride the wind; a packed hillside does not. Ignored unless `mobile`.
    pub windborne: bool,
    /// Whether this material emits light. Glowing cells are picked up by the
    /// renderer's bloom pass, which is what gives lava its halo.
    pub glow: bool,
}

/// The density of an empty cell. It is an ordinary value in the same ordering as
/// everything else, so a material lighter than this floats up through the air on
/// exactly the rule that makes a heavier one fall through it.
pub const AIR_DENSITY: u8 = 20;

/// What a cell turns into when something is next to it.
///
/// Every reaction is written from *one* cell's point of view: "a cell of `actor`
/// that can see a `trigger` next to it becomes `product`". A two-sided reaction
/// is two rules, one for each participant, which is what lets the kernel decide
/// a cell's next material by looking only at its own neighbourhood — no cell
/// ever writes to another, so the whole grid can react at once.
#[derive(Clone, Copy)]
pub struct Rule {
    /// The material this rule applies to.
    pub actor: MaterialId,
    /// The neighbouring material that sets it off.
    pub trigger: MaterialId,
    /// What the actor becomes.
    pub product: MaterialId,
    /// Which neighbours count (see [`Look`]).
    pub look: Look,
    /// Rarity: the reaction fires with probability `1/chance` per tick, so a
    /// larger number makes it creep rather than happen at once. `1` is instant.
    pub chance: u32,
}

/// Which neighbouring cells a [`Rule`] inspects.
///
/// [`crate::kernels::react`] implements all four, so a new material can reach
/// for whichever one it needs without touching the kernel. The built-in rules
/// only use [`Look::Ortho`]; plugins pick any of them by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Look {
    /// The four orthogonal neighbours: up, down, left, right.
    Ortho = 0,
    /// All eight neighbours, diagonals included.
    Around = 1,
    /// The cell directly above.
    Above = 2,
    /// The cell directly below.
    Below = 3,
}

/// ===================== ADD NEW BUILT-IN MATERIALS HERE =====================
/// The position here is the material's id, so keep `Empty` first and do not
/// reorder existing entries (the reaction rules and the key bindings refer to
/// materials by id).
pub fn table() -> Vec<MaterialInfo> {
    vec![
        empty::INFO, // id 0
        sand::INFO,  // id 1
        stone::INFO, // id 2
        water::INFO, // id 3
        lava::INFO,  // id 4
        soil::INFO,  // id 5
    ]
}

/// Every reaction in the world, gathered from the materials that declare them.
/// Order does not matter: the kernel takes the first rule that both matches and
/// wins its dice roll, and no two rules here apply to the same pair.
pub fn rules() -> Vec<Rule> {
    let mut rules = Vec::new();
    rules.extend_from_slice(water::RULES);
    rules.extend_from_slice(lava::RULES);
    rules
}

impl MaterialInfo {
    /// Whether this material appears in the picker, and can therefore be
    /// painted by hand. Everything currently can; a material that only ever
    /// exists because another one produces it would return false here.
    pub fn pickable(&self) -> bool {
        true
    }
}

/// Every material and every rule the world currently knows about: the built-in
/// tables, plus whatever plugins have added since. This is what the host turns
/// into the GPU tables, so a change here followed by
/// [`crate::sim::Simulation::set_tables`] is a change in the world.
///
/// Materials are only ever added or changed in place, never removed, because a
/// material's id is its position and the cells already in the world refer to
/// it by that number.
#[derive(Clone)]
pub struct Registry {
    materials: Vec<MaterialInfo>,
    rules: Vec<Rule>,
}

impl Registry {
    /// The built-in materials and rules, and nothing else.
    pub fn builtin() -> Self {
        Registry {
            materials: table(),
            rules: rules(),
        }
    }

    /// Every material, in id order.
    pub fn materials(&self) -> &[MaterialInfo] {
        &self.materials
    }

    /// Every reaction, in no particular order.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// The material called `name`, ignoring case. Plugins refer to materials
    /// by name, including the built-in ones.
    pub fn find(&self, name: &str) -> Option<MaterialId> {
        self.materials
            .iter()
            .position(|info| info.name.eq_ignore_ascii_case(name))
            .map(|id| id as MaterialId)
    }

    /// Add a material, or if one with the same name already exists, replace its
    /// properties and keep its id. Replacing is what makes dropping a plugin on
    /// the window a second time a reload rather than a duplicate, and it is
    /// also how a plugin can retune a built-in material.
    pub fn add_material(&mut self, info: MaterialInfo) -> Result<MaterialId, String> {
        if let Some(id) = self.find(info.name) {
            self.materials[id as usize] = info;
            return Ok(id);
        }
        if self.materials.len() >= MAX_MATERIALS {
            return Err(format!(
                "no room for '{}': there can only be {MAX_MATERIALS} materials",
                info.name
            ));
        }
        self.materials.push(info);
        Ok((self.materials.len() - 1) as MaterialId)
    }

    /// Add a reaction. A rule for the same actor and trigger replaces the old
    /// one, so the kernel never has two rules competing for one pair.
    pub fn add_rule(&mut self, rule: Rule) {
        match self
            .rules
            .iter_mut()
            .find(|old| old.actor == rule.actor && old.trigger == rule.trigger)
        {
            Some(old) => *old = rule,
            None => self.rules.push(rule),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn powder(name: &'static str) -> MaterialInfo {
        MaterialInfo { name, ..sand::INFO }
    }

    #[test]
    fn the_builtin_ids_match_the_named_constants() {
        let registry = Registry::builtin();
        assert_eq!(registry.find("Empty"), Some(EMPTY));
        assert_eq!(registry.find("sand"), Some(SAND));
        assert_eq!(registry.find("STONE"), Some(STONE));
        assert_eq!(registry.find("Water"), Some(WATER));
        assert_eq!(registry.find("Lava"), Some(LAVA));
        assert_eq!(registry.find("Soil"), Some(SOIL));
        assert_eq!(registry.find("Acid"), None);
    }

    #[test]
    fn a_new_material_gets_the_next_id_and_the_same_name_keeps_it() {
        let mut registry = Registry::builtin();
        let first = registry.materials().len() as MaterialId;
        let ash = registry.add_material(powder("Ash")).unwrap();
        assert_eq!(ash, first);

        let recoloured = MaterialInfo {
            color: [1, 2, 3],
            ..powder("ash")
        };
        assert_eq!(registry.add_material(recoloured).unwrap(), ash);
        assert_eq!(registry.materials().len() as MaterialId, first + 1);
        assert_eq!(registry.materials()[ash as usize].color, [1, 2, 3]);
    }

    #[test]
    fn the_registry_stops_at_the_last_id_a_cell_can_hold() {
        let mut registry = Registry::builtin();
        while registry.materials().len() < MAX_MATERIALS {
            let name: &'static str = format!("m{}", registry.materials().len()).leak();
            registry.add_material(powder(name)).unwrap();
        }
        assert!(registry.add_material(powder("one too many")).is_err());
    }

    #[test]
    fn a_rule_for_the_same_pair_replaces_the_old_one() {
        let mut registry = Registry::builtin();
        let before = registry.rules().len();
        registry.add_rule(Rule {
            actor: WATER,
            trigger: LAVA,
            product: SAND,
            look: Look::Around,
            chance: 3,
        });
        assert_eq!(registry.rules().len(), before);
        let rule = registry
            .rules()
            .iter()
            .find(|r| r.actor == WATER && r.trigger == LAVA)
            .unwrap();
        assert_eq!(rule.product, SAND);
        assert_eq!(rule.look, Look::Around);

        registry.add_rule(Rule {
            actor: SAND,
            trigger: WATER,
            product: SOIL,
            look: Look::Below,
            chance: 1,
        });
        assert_eq!(registry.rules().len(), before + 1);
    }
}
