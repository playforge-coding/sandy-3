//! Water — a runny liquid. Falls, tumbles, and spreads fast to find its level.
//!
//! The one thing water does beyond flowing is quench lava. That is written here
//! as a rule rather than as code: a water cell that can see lava next to it
//! becomes stone. Lava says the mirror image of this in `lava.rs`, and between
//! them the pair turns to rock wherever the two meet.

use super::{LAVA, Look, MaterialInfo, Rule, STONE, WATER};

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Water",
    color: [64, 120, 220],
    jitter: 16,
    // Lighter than lava, so it floats on top of it, and lighter than sand, so
    // sand sinks straight through.
    density: 130,
    mobile: true,
    passable: true,
    liquid: true,
    // Runny: it levels off almost as fast as it can move.
    spread: 230,
    // A pond is not blown about grain by grain. A stiff gust still shoves the
    // surface, because the movement kernel lets wind push any mobile cell; this
    // flag is about the loose, airborne materials that ride a breeze.
    windborne: false,
    glow: false,
    draft: 0,
};

pub const RULES: &[Rule] = &[Rule {
    actor: WATER,
    trigger: LAVA,
    product: STONE,
    look: Look::Ortho,
    chance: 1,
}];
