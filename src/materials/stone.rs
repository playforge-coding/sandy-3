//! Stone — an immovable solid. Never moves, and nothing pushes through it.
//!
//! Copy this file to add another static material (wall, bedrock, …).

use super::MaterialInfo;

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Stone",
    color: [128, 128, 134],
    jitter: 18,
    density: 255,
    // Neither moves nor gets moved: the two flags a solid turns off.
    mobile: false,
    passable: false,
    liquid: false,
    spread: 0,
    windborne: false,
    glow: false,
};
