-- Acid: a runny, faintly glowing liquid that eats through rock.
--
-- This ships built into the game, but it is an ordinary plugin: a copy of it
-- dropped on the window would load the same way. A material is the same kind
-- of thing as the built-in ones in Rust: a row of properties, plus a few rules
-- saying what turns into what when two materials touch.

local acid = sandy.material {
    name = "Acid",
    color = { 120, 230, 60 },
    jitter = 20,
    -- Lighter than water, so it floats on a pond; heavier than air, so it
    -- falls. Sand, being heavier still, sinks straight through it.
    density = 120,
    mobile = true,
    passable = true,
    liquid = true,
    -- Runny, though not quite as runny as water.
    spread = 200,
    windborne = false,
    glow = true,
}

-- Rock next to acid dissolves, a little at a time, and the acid that did it
-- is spent. Both halves are written from the cell's own point of view, the
-- way every rule is: "a cell of X that can see a Y becomes Z".
for _, rock in ipairs { "Stone", "Soil" } do
    sandy.rule { actor = rock, trigger = acid, product = "Empty", look = "around", chance = 6 }
    sandy.rule { actor = acid, trigger = rock, product = "Empty", look = "around", chance = 12 }
end
