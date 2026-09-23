// Spray: a brush that sprinkles the chosen material rather than painting a
// solid disk.
//
// A few grains land at random inside the brush circle every frame the mouse
// is held, so a slow sweep leaves a light dusting and holding still builds up
// a pile. Being a brush rather than a tool, it paints whatever is picked in
// the material list, the eraser included.

const GRAINS_PER_FRAME = 12;

sandy.brush({
    name: "Spray",
    onDrag: (t) => {
        for (let i = 0; i < GRAINS_PER_FRAME; i++) {
            // A random point in the circle. The square root keeps the grains
            // spread evenly rather than bunched at the centre.
            const angle = Math.random() * 2 * Math.PI;
            const distance = Math.sqrt(Math.random()) * t.radius;
            const x = t.x + Math.cos(angle) * distance;
            const y = t.y + Math.sin(angle) * distance;
            sandy.paint(x, y, 0, t.material);
        }
    },
});
