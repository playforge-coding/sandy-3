//! Soil — packed earth, and the material terrain is built from.
//!
//! Unlike sand, soil is a solid: it holds the shape you paint it in, so a
//! hillside stays a hillside instead of avalanching flat. Mechanically it is
//! stone in a different colour, which is rather the point — the two share every
//! flag, and the difference between them is data.

use super::MaterialInfo;

pub const INFO: MaterialInfo = MaterialInfo {
    name: "Soil",
    color: [104, 72, 44],
    jitter: 22,
    density: 255,
    mobile: false,
    passable: false,
    liquid: false,
    spread: 0,
    windborne: false,
    glow: false,
    draft: 0,
};
