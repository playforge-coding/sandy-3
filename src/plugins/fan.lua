-- Fan: a tool that blows a steady updraft wherever the cursor is held.
--
-- The built-in wind tool blows the way the cursor sweeps. This one always
-- blows up, so holding it under a heap lifts the sand off it. A tool is a
-- function the game calls once a frame while the mouse is held, with a table
-- saying where the cursor is (x, y), where it was last frame (px, py), whether
-- this is the first frame of the stroke, and the brush size and material set
-- in the panel.

sandy.tool {
    name = "Fan",
    on_drag = function(t)
        -- The gust is three times the brush, like the wind tool's, and never
        -- smaller than thirty cells, since a tiny one dies before it has
        -- moved anything.
        local radius = math.max(t.brush * 3, 30)
        sandy.wind(t.x, t.y, radius, 0, -4)
    end,
}
