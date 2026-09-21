-- Disk: the plain brush. Paints a solid circle of the chosen material at the
-- chosen size, which is what a falling-sand game's brush has always done.
--
-- This is the brush the panel starts on. It is a script like any other brush,
-- so there is one way to paint rather than a built-in way and a plugin way,
-- and it is the shortest possible example of one: a brush is a function the
-- game calls once a frame while the mouse is held, with a table saying where
-- the cursor is (x, y), where it was last frame (px, py), whether this is the
-- first frame of the stroke, and the radius and material set in the panel.

sandy.brush {
    name = "Disk",
    on_drag = function(t)
        sandy.paint(t.x, t.y, t.radius, t.material)
    end,
}
