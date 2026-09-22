-- The control API: the `sim` table a control script drives the game with.
--
-- A control script runs as a coroutine. Every function here hands what it
-- was asked to the host by yielding it, along with the name of the request,
-- and waits to be resumed with the answer. The host, in src/scripting.rs,
-- does the work against the real simulation and resumes the script with
-- `true` and the results, or `false` and a message, which is raised here as
-- an error pointing at the line of the script that asked. So a script reads
-- as plain, straight-line code, and the Lua state never touches the GPU.
--
-- `sandy` is in scope as well, so a control script can register materials,
-- brushes, tools and worlds exactly as a plugin does.

local yield, pack, unpack = coroutine.yield, table.pack, table.unpack

-- Every `sim` function below calls this in tail position, so its frame
-- stands in for theirs and level 2 is the line of the script that called
-- them.
local function request(op, ...)
    local reply = pack(yield(op, ...))
    if not reply[1] then
        error(reply[2], 2)
    end
    return unpack(reply, 2, reply.n)
end

local sim = { width = sandy.width, height = sandy.height }

-- Time. `step` runs exactly that many ticks, one by default, there and then.
-- `frame` lets that many frames go by, one by default: in the window the
-- frames are real ones, drawn with the world running at the panel's speed,
-- so a script can be watched; headless, a frame is a sixtieth of a second of
-- the world's clock. Either way a paused world does not move.
function sim.step(ticks) return request("step", ticks) end
function sim.frame(frames) return request("frame", frames) end
function sim.pause() return request("pause") end
function sim.resume() return request("resume") end
function sim.paused() return request("paused") end
function sim.speed(multiplier) return request("speed", multiplier) end
function sim.ticks() return request("ticks") end

-- Changing the world. Painting and blowing are what the plain brush and the
-- wind tool do, and `fill` is a rectangle with both corners included.
function sim.paint(x, y, radius, material) return request("paint", x, y, radius, material) end
function sim.fill(x0, y0, x1, y1, material) return request("fill", x0, y0, x1, y1, material) end
function sim.wind(x, y, radius, dvx, dvy) return request("wind", x, y, radius, dvx, dvy) end
function sim.clear() return request("clear") end
function sim.generate(world, seed) return request("generate", world, seed) end

-- Driving a brush or a tool the way the mouse does: one frame of the stroke
-- per point of the path.
function sim.stroke(spec) return request("stroke", spec) end

-- The panel: what the mouse would paint with.
function sim.pick(spec) return request("pick", spec) end

-- Reading the world back. A snapshot is the whole grid and the wind at one
-- moment; `get` and `count` are one question each, at a readback apiece.
function sim.snapshot() return request("snapshot") end
function sim.get(x, y) return request("get", x, y) end
function sim.count(material) return request("count", material) end

-- Files. A screenshot is written before the call returns. A recording runs
-- from `record` to `stop`; headless, `stop` waits for the file too.
function sim.screenshot(path) return request("screenshot", path) end
function sim.record(path) return request("record", path) end
function sim.stop() return request("stop") end
function sim.plugin(path) return request("plugin", path) end

-- What there is.
function sim.materials() return request("materials") end
function sim.brushes() return request("brushes") end
function sim.tools() return request("tools") end
function sim.worlds() return request("worlds") end

-- End the script here, and close the window if there is one.
function sim.quit() return request("quit") end

return sim
