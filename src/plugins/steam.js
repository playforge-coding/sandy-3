// Steam: what water becomes when it meets something hot.
//
// It is a tile like any other: it rises through the air, drifts, rides a
// gust, and thins away after a few seconds. The one thing it does to the
// world around it is warm the air, so a plume of it pushes upwards on the
// wind field and stands in its own updraft, which sets the air above a
// boiling pool moving and lifts loose sand it passes over.
//
// Water boils where it touches fire or lava. The lava half retunes the
// built-in rule, which had the water turn to stone along with the lava: the
// lava still crusts over, and the water now boils off instead. Steam that
// gathers under a stone or soil ceiling condenses back into water and rains
// down again.

const steam = sandy.material({
    name: "Steam",
    color: [218, 224, 230],
    jitter: 24,
    // Lighter than air, so it rises; heavier than fire, so a flame rises
    // through it.
    density: 10,
    mobile: true,
    passable: true,
    // Drifts sideways a little as it climbs, the way a flame does, only less.
    liquid: true,
    spread: 24,
    windborne: true,
    glow: false,
    // Warm: pushes the air it sits in upwards, in hundredths of a cell a tick.
    draft: 50,
});

// Water boils on contact with fire, and with lava, which loads before this
// and had a rule of its own for the pair.
sandy.rule({ actor: "Water", trigger: "Fire", product: steam, look: "around", chance: 1 });
sandy.rule({ actor: "Water", trigger: "Lava", product: steam, look: "ortho", chance: 1 });

// Thins away into the air over a few seconds.
sandy.rule({ actor: steam, trigger: "Empty", product: "Empty", look: "around", chance: 300 });

// Condenses on a cold ceiling and falls back as water.
for (const rock of ["Stone", "Soil"]) {
    sandy.rule({ actor: steam, trigger: rock, product: "Water", look: "above", chance: 40 });
}
