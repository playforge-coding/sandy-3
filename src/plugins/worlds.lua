-- Worlds: the landscapes the World picker offers.
--
-- A world is a plugin like any other. It registers a name and a function,
-- and when that world is picked the function is handed `w`, a blank canvas
-- the size of the grid, and paints the whole landscape into it. Nothing goes
-- to the GPU until the function returns, so a world can be painted a cell at
-- a time without it costing anything.
--
-- `w` carries the seed typed into the panel (`w.seed`) and the grid size
-- (`w.width`, `w.height`), and has four methods: `w:set(x, y, material)`,
-- `w:get(x, y)`, `w:fill(x0, y0, x1, y1, material)` for a rectangle with both
-- corners included, and `w:disk(x, y, radius, material)`. Cells off the grid
-- are ignored. The origin is the top left, so a smaller y is higher up.
--
-- The four worlds here share one generator, `landscape`, that takes a table
-- of knobs: where the ground sits, how hilly it is, what it is made of, and
-- where the sea comes up to. `sandy.noise` gives it the shape of the land,
-- and `math.random`, which is reseeded from the world seed before every
-- generation, scatters the trees, so the same seed always makes the same
-- world.

-- How many cells of the ground's top layer are the surface material; the
-- rest is the material underneath.
local SURFACE_DEPTH = 7

-- Roughly one in this many columns of dry land grows a tree.
local TREE_RARITY = 11

-- A tree: a wood trunk rising from the ground at `x`, capped with a round
-- canopy of leaves. Leaves only go into empty cells, so the trunk stays
-- visible through the middle and two trees can overlap without one eating
-- the other. `surface` is the topmost ground cell of the column.
local function plant_tree(w, x, surface)
    local trunk = math.random(6, 12)
    local radius = math.random(3, 5)
    -- Not if the canopy would not fit under the top of the world.
    if surface <= trunk + radius + 1 then
        return
    end
    w:fill(x, surface - 1, x, surface - trunk, "Wood")
    local cy = surface - trunk - 1
    for dy = -radius, radius do
        for dx = -radius, radius do
            if dx * dx + dy * dy <= radius * radius and w:get(x + dx, cy + dy) == 0 then
                w:set(x + dx, cy + dy, "Leaves")
            end
        end
    end
end

-- A generator for a rolling landscape. Heights in `p` are fractions of the
-- world's height, from the top, so `base = 0.5` puts the ground band halfway
-- down and `sea_level = 1` leaves the world dry.
local function landscape(p)
    return function(w)
        local terrain = sandy.noise { seed = w.seed, frequency = p.frequency, octaves = 4 }
        local base = w.height * p.base
        local amplitude = w.height * p.amplitude
        local sea = math.floor(w.height * p.sea_level)
        -- Leave a little sky at the top and a floor at the bottom.
        local highest, lowest = 4, w.height - SURFACE_DEPTH - 2

        -- The surface height of every column, from one line of noise.
        local surface = {}
        for x = 0, w.width - 1 do
            local y = math.floor(base - terrain:at(x, 0) * amplitude)
            surface[x] = math.min(math.max(y, highest), lowest)
        end

        -- The ground: a cap of the surface material over the rest, and
        -- water in any open air that lies below the waterline.
        for x = 0, w.width - 1 do
            local top = surface[x]
            w:fill(x, top, x, top + SURFACE_DEPTH - 1, p.surface)
            w:fill(x, top + SURFACE_DEPTH, x, w.height - 1, p.subsurface)
            if sea < top then
                w:fill(x, sea, x, top - 1, "Water")
            end
        end

        -- Trees, on dry land only, and clear of the edges so a canopy is
        -- never cut off by the side of the world.
        if p.trees then
            for x = 4, w.width - 5 do
                if surface[x] < sea and math.random(TREE_RARITY) == 1 then
                    plant_tree(w, x, surface[x])
                end
            end
        end
    end
end

-- Rolling hills of soil over stone, water pooled in the valleys, and trees.
sandy.world {
    name = "Forest",
    generate = landscape {
        base = 0.5,
        amplitude = 0.3,
        frequency = 0.008,
        sea_level = 0.55,
        surface = "Soil",
        subsurface = "Stone",
        trees = true,
    },
}

-- Near-flat grassland: dry soil with the odd tree and no water at all.
sandy.world {
    name = "Plains",
    generate = landscape {
        base = 0.62,
        amplitude = 0.03,
        frequency = 0.012,
        sea_level = 1,
        surface = "Soil",
        subsurface = "Stone",
        trees = true,
    },
}

-- A deep sea over a gently rolling bed of sand.
sandy.world {
    name = "Ocean",
    generate = landscape {
        base = 0.9,
        amplitude = 0.05,
        frequency = 0.012,
        sea_level = 0.12,
        surface = "Sand",
        subsurface = "Sand",
        trees = false,
    },
}

-- Arid dunes of sand over stone, bone dry.
sandy.world {
    name = "Desert",
    generate = landscape {
        base = 0.5,
        amplitude = 0.2,
        frequency = 0.006,
        sea_level = 1,
        surface = "Sand",
        subsurface = "Stone",
        trees = false,
    },
}
