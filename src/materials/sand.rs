//! Sand — the classic powder. Falls straight down and tumbles into a pile.
//!
//! Sand declares no rules and no special flags: it is simply mobile, heavier
//! than air and not a liquid, and the movement kernel does the rest. Copy this
//! file to add another powder (dirt, ash, salt, …) — change the numbers and you
//! are done.

use super::MaterialInfo;

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Sand",
    color: [194, 178, 128],
    jitter: 28,
    // Heavier than water and lava, so a poured stream sinks through both.
    density: 180,
    mobile: true,
    passable: true,
    // Not a liquid: it piles at an angle of repose instead of levelling off.
    liquid: false,
    spread: 0,
    // Loose grains: a gust slants a falling stream and drifts the dune it builds.
    windborne: true,
    glow: false,
    draft: 0,
};
