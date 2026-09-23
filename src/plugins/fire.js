// Fire: a flame that rises, licks outwards, and burns out.
//
// There is nothing to burn yet, so fire is what the brush paints: a hot gas,
// lighter than air, that climbs, drifts, and goes out in about half a second.
// It warms the air above it, so a bonfire stands in its own updraft, and it
// boils water into steam (see steam.js) and is put out by it.

const fire = sandy.material({
    name: "Fire",
    color: [255, 140, 40],
    // A lot of per-cell variation, so a flame is mottled rather than flat.
    jitter: 90,
    // Lighter than air, so it rises through it on the same rule that makes
    // sand fall through it.
    density: 5,
    mobile: true,
    passable: true,
    // A gas has no level to find, but the sideways creep a liquid does once
    // it can fall no further is exactly how a flame licks outwards, since a
    // gas that is rising never can fall.
    liquid: true,
    spread: 60,
    // A gust bends it over and carries it.
    windborne: true,
    glow: true,
    // Hot: pushes the air it sits in upwards, in hundredths of a cell a tick.
    draft: 30,
});

// A flame with air next to it burns out, one tick in thirty, so it lasts
// about half a second and a blob of it thins from the edges in.
sandy.rule({ actor: fire, trigger: "Empty", product: "Empty", look: "around", chance: 30 });

// Water puts it out on contact. What that does to the water is in steam.js.
sandy.rule({ actor: fire, trigger: "Water", product: "Empty", look: "around", chance: 1 });
