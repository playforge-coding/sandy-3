//! Lava — a sluggish, glowing liquid.
//!
//! Same motion as water with a much lower `spread`, so it creeps rather than
//! fans out and settles into blobs. It is quenched by water on contact: see the
//! matching rule in `water.rs` for the other half of that reaction.

use super::{LAVA, Look, MaterialInfo, Rule, STONE, WATER};

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Lava",
    color: [207, 70, 24],
    jitter: 36,
    // Denser than water so it sinks below it, lighter than sand so sand still
    // sinks through.
    density: 160,
    mobile: true,
    passable: true,
    liquid: true,
    // Viscous: it barely creeps sideways.
    spread: 40,
    windborne: false,
    // Molten: flagged for the renderer's bloom pass, which gives it its halo.
    glow: true,
};

pub const RULES: &[Rule] = &[Rule {
    actor: LAVA,
    trigger: WATER,
    product: STONE,
    look: Look::Ortho,
    chance: 1,
}];
