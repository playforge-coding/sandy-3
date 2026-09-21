-- Spray: sprinkles the chosen material rather than painting a solid disk.
--
-- A few grains land at random inside the brush circle every frame the mouse
-- is held, so a slow sweep leaves a light dusting and holding still builds up
-- a pile. Whatever is picked in the material list is what comes out.

local GRAINS_PER_FRAME = 12

sandy.tool {
    name = "Spray",
    on_drag = function(t)
        for _ = 1, GRAINS_PER_FRAME do
            -- A random point in the circle. The square root keeps the grains
            -- spread evenly rather than bunched at the centre.
            local angle = math.random() * 2 * math.pi
            local distance = math.sqrt(math.random()) * t.brush
            local x = t.x + math.cos(angle) * distance
            local y = t.y + math.sin(angle) * distance
            sandy.paint(x, y, 0, t.material)
        end
    end,
}
