//! Empty — air / nothing.
//!
//! It has no behaviour of its own, but it is a material like any other: it has a
//! density, so a heavier cell sinks through it and a lighter one rises past it
//! without either case needing a special rule in the movement kernel. Its colour
//! is the daytime sky the whole world is drawn against.

use super::{AIR_DENSITY, MaterialInfo};

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Empty",
    color: [124, 173, 222],
    jitter: 0,
    density: AIR_DENSITY,
    // Air never moves on its own; it only ever gets swapped out of the way by
    // something that does.
    mobile: false,
    passable: true,
    liquid: false,
    spread: 0,
    windborne: false,
    glow: false,
    draft: 0,
};
