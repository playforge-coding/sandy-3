// Wood and leaves: what a tree is made of, and the first things in the game
// that burn.
//
// Both are solids, like stone and soil, so a tree stands where the world
// generator put it. What sets them apart is a pair of rules: a cell of either
// that can see fire, or lava, catches and becomes fire itself. Leaves catch
// in a moment and wood takes its time, so a spark in the canopy runs through
// it and then works slowly down the trunk. The fire a burning cell turns
// into is the ordinary fire from fire.js: it rises, licks along the
// underside of whatever is above it, and burns out when it reaches open air,
// which is what keeps a burning tree from setting light to the whole sky.

sandy.material({
    name: "Wood",
    color: [96, 64, 36],
    jitter: 16,
    density: 255,
    mobile: false,
    passable: false,
});

sandy.material({
    name: "Leaves",
    color: [58, 132, 56],
    jitter: 30,
    density: 255,
    mobile: false,
    passable: false,
});

// Both catch from fire and from lava. The chance is one in how many ticks
// while the heat is next to the cell.
for (const hot of ["Fire", "Lava"]) {
    sandy.rule({ actor: "Wood", trigger: hot, product: "Fire", look: "around", chance: 12 });
    sandy.rule({ actor: "Leaves", trigger: hot, product: "Fire", look: "around", chance: 3 });
}
