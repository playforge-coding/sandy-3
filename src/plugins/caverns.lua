-- Caverns: solid rock riddled with hollows, water pooled in the deeper ones
-- and lava in the deepest.
--
-- Where worlds.lua shapes the land from one line of noise, this one uses a
-- whole sheet of it. `noise:grid(width, height)` asks FastNoise2 for every
-- cell at once, which it does in a few milliseconds, and hands back a table
-- of rows, `field[y][x]`, both counted from zero. Wherever the noise is high
-- enough the rock is hollowed out, so the caves are the blobs of a fractal
-- field and join up the way such blobs do.

-- Above this the world is open sky; the rock starts here.
local ROOF = 0.2

-- A cell of rock this far into the noise's range is hollowed out. Higher is
-- fewer, smaller caves.
local HOLLOW = 0.25

-- The middle of a pocket, where the noise is highest, holds what has seeped
-- into it: water in the lower half of the world, lava near the bottom.
local POOL = 0.45
local WATER_FROM = 0.6
local LAVA_FROM = 0.88

sandy.world {
    name = "Caverns",
    generate = function(w)
        local rock = sandy.noise { seed = w.seed, frequency = 0.012, octaves = 3 }
        local field = rock:grid(w.width, w.height)
        local roof = math.floor(w.height * ROOF)

        w:fill(0, roof, w.width - 1, w.height - 1, "Stone")
        for y = roof, w.height - 1 do
            local row = field[y]
            local depth = y / w.height
            for x = 0, w.width - 1 do
                local n = row[x]
                if n > HOLLOW then
                    if n > POOL and depth > LAVA_FROM then
                        w:set(x, y, "Lava")
                    elseif n > POOL and depth > WATER_FROM then
                        w:set(x, y, "Water")
                    else
                        w:set(x, y, "Empty")
                    end
                end
            end
        end
    end,
}
