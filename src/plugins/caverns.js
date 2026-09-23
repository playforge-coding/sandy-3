// Caverns: solid rock riddled with hollows, water pooled in the deeper ones
// and lava in the deepest.
//
// Where worlds.js shapes the land from one line of noise, this one uses a
// whole sheet of it. `noise.grid(width, height)` asks FastNoise2 for every
// cell at once, which it does in a few milliseconds, and hands back an array
// of rows, `field[y][x]`, both counted from zero. Wherever the noise is high
// enough the rock is hollowed out, so the caves are the blobs of a fractal
// field and join up the way such blobs do.

// Above this the world is open sky; the rock starts here.
const ROOF = 0.2;

// A cell of rock this far into the noise's range is hollowed out. Higher is
// fewer, smaller caves.
const HOLLOW = 0.25;

// The middle of a pocket, where the noise is highest, holds what has seeped
// into it: water in the lower half of the world, lava near the bottom.
const POOL = 0.45;
const WATER_FROM = 0.6;
const LAVA_FROM = 0.88;

sandy.world({
    name: "Caverns",
    generate: (w) => {
        const rock = sandy.noise({ seed: w.seed, frequency: 0.012, octaves: 3 });
        const field = rock.grid(w.width, w.height);
        const roof = Math.floor(w.height * ROOF);

        w.fill(0, roof, w.width - 1, w.height - 1, "Stone");
        for (let y = roof; y < w.height; y++) {
            const row = field[y];
            const depth = y / w.height;
            for (let x = 0; x < w.width; x++) {
                const n = row[x];
                if (n > HOLLOW) {
                    if (n > POOL && depth > LAVA_FROM) {
                        w.set(x, y, "Lava");
                    } else if (n > POOL && depth > WATER_FROM) {
                        w.set(x, y, "Water");
                    } else {
                        w.set(x, y, "Empty");
                    }
                }
            }
        }
    },
});
