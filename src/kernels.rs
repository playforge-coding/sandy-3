//! The simulation itself, as GPU compute kernels.
//!
//! Every kernel here is an ordinary looking Rust function with `#[kernel]` on
//! it. [unipute](https://github.com/playforge-coding/unipute) turns each one
//! into WGSL while this crate compiles, so a mistake in a rule is a build error
//! pointing at the line that caused it rather than a shader compile failure at
//! startup. The host side in [`crate::sim`] reads the bindings and the workgroup
//! size straight off the generated types, so adding a parameter to a kernel does
//! not need a matching edit over there.
//!
//! # What runs when
//!
//! One tick is the wind first ([`curl`], [`swirl`], [`flow`], [`divergence`],
//! [`pressure`] a few times over, [`project`]), then [`react`] once, then
//! [`movement`] [`MOVE_PASSES`] times. Reacting and moving are separate because they need
//! the grid in different shapes: a reaction reads a neighbourhood and rewrites a
//! single cell, while a move has to pick two cells up and put them down
//! somewhere else. Splitting them keeps each kernel to one job.
//!
//! # The wind is a fluid
//!
//! The wind is a velocity field with one vector per cell, in cells per tick.
//! The wind tool stamps velocity into it, and every tick it is moved along by
//! itself and squeezed until it is incompressible, which is what turns a
//! stamped puff into a gust that travels across the world, spreads out where it
//! meets a wall, and curls into eddies at its edges. That is the same scheme
//! sandspiel's wind uses, in six small kernels: [`curl`] and [`swirl`] keep
//! the eddies wound up, [`flow`] carries the field along and lets it fade,
//! [`divergence`] measures where it is piling up, [`pressure`] solves for the
//! push that would stop that, and [`project`] applies it. [`movement`] then
//! reads the finished field to decide what gets shoved.
//!
//! Both read one buffer and write another, never the same one, which is what
//! makes them safe to run over the whole grid at once: no invocation can see a
//! half-finished neighbour, so the result does not depend on the order the GPU
//! happens to schedule them in. The host bounces the two buffers back and forth.
//!
//! # How movement avoids fighting itself
//!
//! A CPU falling-sand engine scans the grid bottom to top and moves one cell at
//! a time, which is exactly what a GPU cannot do. [`movement`] instead cuts the
//! grid into two-by-two blocks and gives one invocation the whole block: it
//! reads four cells and writes those same four, so two invocations never argue
//! over a cell and no locking is needed. The cut is shifted by one cell on
//! alternate passes, so a cell that sat at a block edge one pass sits in the
//! middle of one the next and nothing gets stuck on the seams. That is a
//! Margolus neighbourhood, and it is the usual way to run a cellular automaton
//! in parallel.
//!
//! A block can only move a cell as far as its own corner, so one pass moves
//! anything at most one cell. [`MOVE_PASSES`] passes per tick is what sets how
//! fast things fall.
//!
//! # The tables
//!
//! Neither kernel names a material. What sand and water *are* lives in two
//! tables the host uploads from [`crate::materials`]:
//!
//! - **props**, [`PROPS_STRIDE`] words per material: density, flags, packed
//!   colour, spread. [`movement`] reads it to decide what sinks through what.
//! - **rules**, [`RULE_STRIDE`] words per rule: actor, trigger, product, look,
//!   chance. [`react`] walks it.
//!
//! So a new material is a new row in each table, and neither kernel changes.

use unipute::kernel;

use crate::materials::{MAX_MATERIALS, MaterialInfo, Registry, Rule};

/// Words per material in the props table. A power of two, so the kernels reach
/// a material's row with a shift rather than a multiply.
pub const PROPS_STRIDE: usize = 8;

/// Words per rule in the reaction table.
pub const RULE_STRIDE: usize = 8;

/// The flag bits packed into a material's props word 1. The kernels test these
/// by value, so the numbers here and the literals over there have to agree.
pub const FLAG_MOBILE: u32 = 1;
pub const FLAG_PASSABLE: u32 = 2;
pub const FLAG_LIQUID: u32 = 4;
pub const FLAG_WINDBORNE: u32 = 8;
/// Read by the renderer rather than by a kernel: a glowing cell is picked up by
/// the bloom pass.
pub const FLAG_GLOW: u32 = 16;

/// How many [`movement`] passes make up one tick. Each pass moves a cell at most
/// one place, so this is roughly how many cells a grain falls per tick. An odd
/// number means the pair of grid buffers ends a tick the way it started, once
/// [`react`]'s own pass is counted, so the live grid is always the same buffer
/// between ticks.
pub const MOVE_PASSES: usize = 3;

/// How many [`pressure`] passes go into making the wind incompressible each
/// tick. More is more exact; the solve starts from the previous tick's answer,
/// so a few passes a tick catch up over a handful of ticks. Even, so the answer
/// ends up back in the buffer it started in.
pub const PRESSURE_ITERATIONS: usize = 20;

/// How much of the last tick's pressure the solve starts from. All of it would
/// be the best guess while a gust blows, but the solve is far too slow to
/// forget the broad shape of a pressure once the gust that made it has gone,
/// and a stale pressure keeps pushing the air about for good. Starting from
/// most of it keeps the guess useful and lets the rest fade in a few ticks.
pub const PRESSURE_CARRY: f32 = 0.9;

/// The fastest the wind blows, in cells per tick. [`gust`] clamps to this, so a
/// flick of the tool is a strong gust rather than a hurricane.
pub const WIND_MAX: f32 = 6.0;

/// What is left of the wind after a tick. [`flow`] multiplies by this. It is
/// close to one because the grid loses plenty on its own: a gust dies away over
/// a few seconds rather than blowing for ever.
pub const WIND_DECAY: f32 = 0.995;

/// How hard [`swirl`] winds the eddies up each tick, in cells per tick of push
/// per unit of spin. Enough to keep a gust curling; too much and every breath
/// of air knots itself into a whirlwind.
pub const SWIRL: f32 = 0.2;

/// The numbers above, packed the way the kernels that read them take them: the
/// top speed, the decay, the swirl, and a spare.
pub fn weather() -> [f32; 4] {
    [WIND_MAX, WIND_DECAY, SWIRL, 0.0]
}

/// Peak strength of the prevailing breeze, in cells per tick. Kept well under
/// the speed a settled grain needs before it moves, so it leans a falling stream
/// of sand without disturbing anything that has landed.
pub const AMBIENT_MAX: f32 = 0.35;

/// How fast the prevailing breeze swings, in radians per tick. A full
/// reverse-and-back takes `2 * PI` over this, which is about twenty seconds.
pub const AMBIENT_RATE: f32 = 0.0026;

/// The props table the kernels index, in material id order.
///
/// The table always has a row for every id a cell could hold, whether or not a
/// material has claimed it yet, so that a plugin adding one later is a write
/// into the same buffer rather than a new buffer and new bind groups. An
/// unclaimed row is all zeros, and nothing in the world ever refers to one.
///
/// One extra row is appended at the end: the *wall* a block uses for a corner
/// that falls outside the world. It is neither mobile nor passable, so nothing
/// moves into it and nothing moves out, which is how the edge of the world holds
/// without either kernel checking for it.
pub fn props_table(registry: &Registry) -> Vec<u32> {
    let mut words = Vec::with_capacity((MAX_MATERIALS + 1) * PROPS_STRIDE);
    for info in registry.materials() {
        words.extend_from_slice(&props_row(info));
    }
    words.resize(MAX_MATERIALS * PROPS_STRIDE, 0);
    // The out-of-world wall. Heavy, and with no flags at all.
    let mut wall = [0u32; PROPS_STRIDE];
    wall[0] = 255;
    words.extend_from_slice(&wall);
    words
}

fn props_row(info: &MaterialInfo) -> [u32; PROPS_STRIDE] {
    let mut flags = 0;
    if info.mobile {
        flags |= FLAG_MOBILE;
    }
    if info.passable {
        flags |= FLAG_PASSABLE;
    }
    if info.liquid {
        flags |= FLAG_LIQUID;
    }
    if info.windborne {
        flags |= FLAG_WINDBORNE;
    }
    if info.glow {
        flags |= FLAG_GLOW;
    }
    let [r, g, b] = info.color;
    let color = r as u32 | (g as u32) << 8 | (b as u32) << 16 | (info.jitter as u32) << 24;
    let mut row = [0u32; PROPS_STRIDE];
    row[0] = info.density as u32;
    row[1] = flags;
    row[2] = color;
    row[3] = info.spread as u32;
    row
}

/// The reaction table [`react`] walks.
///
/// A world with no reactions in it would leave the buffer empty, which wgpu will
/// not bind, so an unreachable row is added in that case. It costs one comparison
/// a tick and saves the kernel a special case.
pub fn rules_table(registry: &Registry) -> Vec<u32> {
    let rules = registry.rules();
    let mut words = Vec::new();
    for rule in rules {
        words.extend_from_slice(&rule_row(rule));
    }
    if rules.is_empty() {
        // An actor id no material has, so the row never matches.
        let mut dead = [0u32; RULE_STRIDE];
        dead[0] = u32::MAX;
        words.extend_from_slice(&dead);
    }
    words
}

fn rule_row(rule: &Rule) -> [u32; RULE_STRIDE] {
    let mut row = [0u32; RULE_STRIDE];
    row[0] = rule.actor as u32;
    row[1] = rule.trigger as u32;
    row[2] = rule.product as u32;
    row[3] = rule.look as u32;
    row[4] = rule.chance.max(1);
    row
}

// ---------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------
//
// A kernel can only call functions declared inside its own body, so the hash
// below is written out once per kernel that needs randomness. It is the same
// function each time, and deliberately so: keeping it inline is what lets the
// macro see the whole kernel.

/// Turn every cell that has something to react to into whatever it reacts into.
///
/// Reactions are written from one cell's point of view: a cell looks around,
/// finds a rule whose actor is its own material and whose trigger is next to it,
/// and becomes that rule's product. Nothing writes to a neighbour, so every cell
/// in the world can be decided at the same time from the same snapshot. A
/// two-sided reaction, like water and lava both turning to stone where they
/// meet, is two rules that happen to fire on the same tick.
///
/// `world` is the width, the height, the tick counter, and a spare.
#[kernel(workgroup_size(16, 16))]
pub fn react(src: &[u32], dst: &mut [u32], rules: &[u32], world: &Vec4<u32>) {
    /// A cheap integer hash. Two cells that differ anywhere, including in the
    /// tick they are being asked about, come out uncorrelated.
    fn hash(seed: u32) -> u32 {
        let mut v = seed;
        v ^= v >> 16u32;
        v *= 2246822519u32;
        v ^= v >> 13u32;
        v *= 3266489917u32;
        v ^= v >> 16u32;
        v
    }

    let width = world.x;
    let height = world.y;
    let frame = world.z;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }

    let index = y * width + x;
    let cell = src[index];
    let material = cell & 255u32;
    // Nothing happening is the common case, so say so up front and only
    // overwrite it if a rule fires.
    dst[index] = cell;

    let count = rules.len() / 8u32;
    let mut rule = 0u32;
    while rule < count {
        let row = rule * 8u32;
        if rules[row] == material {
            // Roll the dice before looking around. A rule that has lost its roll
            // cannot fire whatever the neighbours say, so this skips the reads.
            let chance = rules[row + 4u32];
            let roll = hash(index * 2654435761u32 + frame * 374761393u32 + rule);
            if roll % chance == 0u32 {
                let trigger = rules[row + 1u32];
                let look = rules[row + 3u32];
                let mut found = 0u32;

                for oy in 0..3u32 {
                    for ox in 0..3u32 {
                        // A cell is not its own neighbour.
                        if ox == 1u32 && oy == 1u32 {
                            continue;
                        }
                        // Does this offset count, for the way this rule looks?
                        let mut counts = 0u32;
                        if look == 1u32 {
                            // Around: all eight.
                            counts = 1u32;
                        }
                        if look == 0u32 && (ox == 1u32 || oy == 1u32) {
                            // Ortho: shares a row or a column with us.
                            counts = 1u32;
                        }
                        if look == 2u32 && ox == 1u32 && oy == 0u32 {
                            // Above.
                            counts = 1u32;
                        }
                        if look == 3u32 && ox == 1u32 && oy == 2u32 {
                            // Below.
                            counts = 1u32;
                        }

                        if counts == 1u32 {
                            let nx = x as i32 + ox as i32 - 1;
                            let ny = y as i32 + oy as i32 - 1;
                            if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                                let neighbour = src[ny as u32 * width + nx as u32];
                                if (neighbour & 255u32) == trigger {
                                    found = 1u32;
                                }
                            }
                        }
                    }
                }

                if found == 1u32 {
                    let variant = hash(index + frame * 747796405u32) & 255u32;
                    dst[index] = rules[row + 2u32] | (variant << 8u32);
                    return;
                }
            }
        }
        rule += 1u32;
    }
}

/// Move everything that has somewhere to go, one two-by-two block at a time.
///
/// Each invocation owns one block: it reads the four cells, shuffles them among
/// themselves, and writes all four back. Because no two blocks share a cell,
/// the whole grid moves at once without any of it needing to agree on an order.
/// `world.w` says which pass of the tick this is; its lowest bit shifts the
/// block grid by one cell so the seams land somewhere different each pass.
///
/// Four things can move a cell, and they are tried in this order: it falls, it
/// tumbles off a pile, a liquid creeps sideways to find its level, and the wind
/// shoves it. Which of them applies to a given material is decided entirely by
/// the props table. A strong enough updraft also stops a cell from falling in
/// the first place, which is what lets a gust pick sand up off the ground and
/// carry it rather than only nudge it along.
///
/// The source buffer is only read, so the block is free to peek at cells outside
/// itself. It does that to ask whether a cell has anywhere to fall, which is
/// what stops a liquid from spreading sideways while it is still in mid air.
///
/// `wind` is the velocity field the fluid kernels leave behind, one vector per
/// cell in cells per tick, and `breeze` is the prevailing wind added to every
/// cell's sideways component on top of it.
#[kernel(workgroup_size(8, 8))]
pub fn movement(
    src: &[u32],
    dst: &mut [u32],
    props: &[u32],
    wind: &[Vec2<f32>],
    world: &Vec4<u32>,
    breeze: &f32,
) {
    /// A cheap integer hash, the same one [`react`] uses.
    fn hash(seed: u32) -> u32 {
        let mut v = seed;
        v ^= v >> 16u32;
        v *= 2246822519u32;
        v ^= v >> 13u32;
        v *= 3266489917u32;
        v ^= v >> 16u32;
        v
    }

    /// Whether a heavier cell sitting above a lighter one should trade places.
    ///
    /// This one test is the whole of gravity here. Either the heavy one sinks,
    /// which needs it to move under its own weight and needs the light one to
    /// give way, or the light one floats up past it, which needs the same two
    /// things of the other cell. Sand falling through air, sand sinking through
    /// water and water floating on lava are all this, with different numbers.
    fn trades(top_density: u32, top_flags: u32, low_density: u32, low_flags: u32) -> u32 {
        if top_density <= low_density {
            return 0u32;
        }
        if (top_flags & 1u32) != 0u32 && (low_flags & 2u32) != 0u32 {
            return 1u32;
        }
        if (low_flags & 1u32) != 0u32 && (top_flags & 2u32) != 0u32 {
            return 1u32;
        }
        0u32
    }

    /// How much wind a cell feels beyond what it takes to move it, in cells per
    /// tick. Zero or less means the wind is not enough.
    ///
    /// Loose material already in the air goes with the lightest breeze, which is
    /// what leans a falling stream. The same material once it has landed needs
    /// a real gust, which is what keeps a dune from blowing apart in the day's
    /// weather. Anything else that moves at all, water say, needs a stiff one,
    /// and only ever gets its surface ruffled. `freedom` is whether the cell
    /// could fall straight down, which is what tells the airborne from the
    /// settled.
    fn urge(flags: u32, freedom: u32, wind: f32) -> f32 {
        let mut threshold = 2.0;
        if (flags & 8u32) != 0u32 {
            threshold = 0.5;
        }
        if (flags & 8u32) != 0u32 && freedom != 0u32 {
            threshold = 0.0;
        }
        wind - threshold
    }

    /// Whether an updraft holds a cell up this pass instead of letting it fall.
    ///
    /// The odds climb with the wind, and reach certainty one cell per tick past
    /// what it takes to move the cell at all, so a gust that is strong enough
    /// does not merely slow sand down but lifts it clean off the ground.
    fn held(flags: u32, freedom: u32, lift: f32, roll: u32) -> u32 {
        if (flags & 1u32) == 0u32 {
            return 0u32;
        }
        let excess = urge(flags, freedom, lift);
        if excess > 0.0 && (roll as f32) < excess * 256.0 {
            return 1u32;
        }
        0u32
    }

    /// Whether a cell should slide one place into its neighbour, sideways or
    /// against gravity.
    ///
    /// Two quite different things end up here, because they need the same room
    /// to happen: a liquid levelling itself out, and a gust shoving something
    /// downwind. `spread` is the liquid's runniness and is passed as zero for a
    /// vertical pair, where only the wind clause can fire. `freedom` is whether
    /// the cell could have fallen straight down instead, since a liquid only
    /// creeps once it has landed and loose material only rides a breeze while it
    /// is still in the air. `downwind` is the wind blowing from this cell
    /// towards its neighbour, in cells per tick.
    fn slides(
        from_density: u32,
        from_flags: u32,
        spread: u32,
        freedom: u32,
        into_density: u32,
        into_flags: u32,
        downwind: f32,
        roll: u32,
    ) -> u32 {
        // Room to move: it has to move under its own weight at all, and where it
        // is going has to give way and be lighter than it.
        if (from_flags & 1u32) == 0u32 {
            return 0u32;
        }
        if (into_flags & 2u32) == 0u32 {
            return 0u32;
        }
        if from_density <= into_density {
            return 0u32;
        }

        // A liquid with nowhere left to fall creeps sideways to find its level.
        if (from_flags & 4u32) != 0u32 && freedom == 0u32 && roll < spread {
            return 1u32;
        }

        // The wind. A pass moves a cell one place at most and there are three
        // passes a tick, so odds of a third per cell per tick of wind is what
        // makes a grain in the air travel at the speed of the air around it.
        let excess = urge(from_flags, freedom, downwind);
        if excess > 0.0 && (roll as f32) < excess * 85.0 {
            return 1u32;
        }
        0u32
    }

    let width = world.x;
    let height = world.y;
    let frame = world.z;
    let pass = world.w;

    // The block this invocation owns. Odd passes shift it by one cell so the
    // lines it cuts the grid along are not the same twice running.
    let phase = (pass & 1u32) as i32;
    let x0 = (global_id().x * 2u32) as i32 - phase;
    let y0 = (global_id().y * 2u32) as i32 - phase;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Which of the block's rows and columns are inside the world. The shift puts
    // one row and one column outside it at each edge on every other pass.
    let mut left_in = 0u32;
    if x0 >= 0 && x0 < width as i32 {
        left_in = 1u32;
    }
    let mut right_in = 0u32;
    if x1 >= 0 && x1 < width as i32 {
        right_in = 1u32;
    }
    let mut top_in = 0u32;
    if y0 >= 0 && y0 < height as i32 {
        top_in = 1u32;
    }
    let mut low_in = 0u32;
    if y1 >= 0 && y1 < height as i32 {
        low_in = 1u32;
    }
    if left_in + right_in == 0u32 {
        return;
    }
    if top_in + low_in == 0u32 {
        return;
    }

    // Indices for the four corners. Clamped, so the arithmetic is always in
    // bounds even for a corner that is outside the world; the flags above are
    // what decide whether a corner is really there.
    let lx = clamp(x0, 0, width as i32 - 1) as u32;
    let rx = clamp(x1, 0, width as i32 - 1) as u32;
    let ty = clamp(y0, 0, height as i32 - 1) as u32;
    let by = clamp(y1, 0, height as i32 - 1) as u32;
    let index_a = ty * width + lx;
    let index_b = ty * width + rx;
    let index_c = by * width + lx;
    let index_d = by * width + rx;

    // The four cells, laid out as
    //
    //     a b
    //     c d
    //
    // alongside the row each one's material has in the props table. A corner
    // outside the world reads as the wall row at the end of the table, so it
    // neither moves nor lets anything through.
    let wall = props.len() - 8u32;
    let mut a = 0u32;
    let mut pa = wall;
    if left_in == 1u32 && top_in == 1u32 {
        a = src[index_a];
        pa = (a & 255u32) * 8u32;
    }
    let mut b = 0u32;
    let mut pb = wall;
    if right_in == 1u32 && top_in == 1u32 {
        b = src[index_b];
        pb = (b & 255u32) * 8u32;
    }
    let mut c = 0u32;
    let mut pc = wall;
    if left_in == 1u32 && low_in == 1u32 {
        c = src[index_c];
        pc = (c & 255u32) * 8u32;
    }
    let mut d = 0u32;
    let mut pd = wall;
    if right_in == 1u32 && low_in == 1u32 {
        d = src[index_d];
        pd = (d & 255u32) * 8u32;
    }

    // Can each corner fall straight down? One bit each. This looks at the source
    // buffer, which nothing is writing to, so a block is free to read past its
    // own edge here, which matters because the cell under the bottom row belongs
    // to the next block along.
    //
    // It is the one thing the rest of the pass needs to know about a cell beyond
    // its own material. A liquid with room left below it has no business
    // spreading sideways instead of falling, a grain that has settled has no
    // business riding the breeze, and this is what tells those apart from the
    // ones in mid air.
    let mut freedom = 0u32;
    for corner in 0..4u32 {
        let cx = x0 + (corner & 1u32) as i32;
        let cy = y0 + ((corner >> 1u32) & 1u32) as i32;
        if cx >= 0 && cy >= 0 && cx < width as i32 && cy + 1 < height as i32 {
            let here = src[cy as u32 * width + cx as u32];
            let ph = (here & 255u32) * 8u32;
            let under = src[(cy + 1) as u32 * width + cx as u32];
            let pu = (under & 255u32) * 8u32;
            if trades(props[ph], props[ph + 1u32], props[pu], props[pu + 1u32]) == 1u32 {
                freedom |= 1u32 << corner;
            }
        }
    }
    // These travel with the cell from here on, not with the corner it started
    // in. A drop of water that falls from the top of the block to the bottom of
    // it still has open air under the block, and carrying its answer down with
    // it is what stops it from also creeping sideways on the same pass, halfway
    // through the air. Every swap below therefore moves three things: the cell,
    // the row its material has in the props table, and this.
    let mut free_a = freedom & 1u32;
    let mut free_b = (freedom >> 1u32) & 1u32;
    let mut free_c = (freedom >> 2u32) & 1u32;
    let mut free_d = (freedom >> 3u32) & 1u32;

    // One roll per decision the block makes, all cut from a single hash.
    let dice = hash(
        (global_id().y * 65536u32 + global_id().x) * 2654435761u32
            + frame * 374761393u32
            + pass * 668265263u32,
    );
    let roll_ab = dice & 255u32;
    let roll_cd = (dice >> 8u32) & 255u32;
    let roll_ac = (dice >> 16u32) & 255u32;
    let roll_bd = (dice >> 24u32) & 255u32;
    let flip = (dice >> 7u32) & 1u32;

    // The wind at the block: the day's prevailing breeze plus whatever the
    // fluid is doing here. The fluid is stilled inside anything solid and
    // slowed inside sand and water, so the block takes the strongest of its
    // four corners: a grain on the surface of a heap then feels the air
    // blowing over it rather than the calm inside the pile it sits on. Up is
    // negative, since rows count downwards, so the lift is the vertical
    // component the other way up.
    let mut air = wind[index_a];
    let mut strongest = air.x * air.x + air.y * air.y;
    let corner_b = wind[index_b];
    let speed_b = corner_b.x * corner_b.x + corner_b.y * corner_b.y;
    if speed_b > strongest {
        air = corner_b;
        strongest = speed_b;
    }
    let corner_c = wind[index_c];
    let speed_c = corner_c.x * corner_c.x + corner_c.y * corner_c.y;
    if speed_c > strongest {
        air = corner_c;
        strongest = speed_c;
    }
    let corner_d = wind[index_d];
    let speed_d = corner_d.x * corner_d.x + corner_d.y * corner_d.y;
    if speed_d > strongest {
        air = corner_d;
    }
    let wind_x = breeze + air.x;
    let wind_y = air.y;
    let lift = 0.0 - wind_y;

    // A gust does not only push: blowing along a surface it kicks loose grains
    // up off it, and whatever it has picked up it keeps aloft for as long as it
    // blows hard, the way a dust storm carries sand rather than rolling it. So
    // most of the sideways wind counts as lift for holding a cell up, and for
    // getting one off the ground, though not for carrying one that is already
    // in the air any higher.
    let hold = lift + abs(wind_x) * 0.8;

    // An updraft strong enough holds a cell up. That is decided before gravity
    // gets a look in, and it stops the tumble too, since a grain that is being
    // carried is not resting on anything to roll off. Only the top cell of a
    // column can be held: it is the one that would fall.
    let held_left = held(props[pa + 1u32], free_a, hold, roll_ac);
    let held_right = held(props[pb + 1u32], free_b, hold, roll_bd);

    // Gravity, one column at a time. Whether a column moved is remembered,
    // because a grain that has just fallen has had its move for this pass and
    // must not then also roll off to the side.
    let mut fell_left = 0u32;
    if held_left == 0u32 && trades(props[pa], props[pa + 1u32], props[pc], props[pc + 1u32]) == 1u32
    {
        let cell = a;
        a = c;
        c = cell;
        let row = pa;
        pa = pc;
        pc = row;
        let fall = free_a;
        free_a = free_c;
        free_c = fall;
        fell_left = 1u32;
    }
    let mut fell_right = 0u32;
    if held_right == 0u32
        && trades(props[pb], props[pb + 1u32], props[pd], props[pd + 1u32]) == 1u32
    {
        let cell = b;
        b = d;
        d = cell;
        let row = pb;
        pb = pd;
        pd = row;
        let fall = free_b;
        free_b = free_d;
        free_d = fall;
        fell_right = 1u32;
    }

    // Tumbling: a grain that could not go straight down rolls off to one side
    // instead, which is what stops a pile from stacking into a tower and gives
    // it its slope. Which diagonal is tried first is down to the dice, or every
    // pile would lean the same way.
    for turn in 0..2u32 {
        if ((turn + flip) & 1u32) == 0u32 {
            // a rolls down to the right.
            if held_left == 0u32
                && fell_left == 0u32
                && trades(props[pa], props[pa + 1u32], props[pd], props[pd + 1u32]) == 1u32
            {
                let cell = a;
                a = d;
                d = cell;
                let row = pa;
                pa = pd;
                pd = row;
                let fall = free_a;
                free_a = free_d;
                free_d = fall;
            }
        } else {
            // b rolls down to the left.
            if held_right == 0u32
                && fell_right == 0u32
                && trades(props[pb], props[pb + 1u32], props[pc], props[pc + 1u32]) == 1u32
            {
                let cell = b;
                b = c;
                c = cell;
                let row = pb;
                pb = pc;
                pc = row;
                let fall = free_b;
                free_b = free_c;
                free_c = fall;
            }
        }
    }

    // Sideways: a liquid finding its level, and the wind.
    //
    // Top row. Only one of the two directions can ever apply, since the test
    // needs the cell that moves to be the heavier of the pair, but they are
    // written as one choice so that a trade cannot be undone by the next line.
    if slides(
        props[pa],
        props[pa + 1u32],
        props[pa + 3u32],
        free_a,
        props[pb],
        props[pb + 1u32],
        wind_x,
        roll_ab,
    ) == 1u32
    {
        let cell = a;
        a = b;
        b = cell;
        let row = pa;
        pa = pb;
        pb = row;
        let fall = free_a;
        free_a = free_b;
        free_b = fall;
    } else if slides(
        props[pb],
        props[pb + 1u32],
        props[pb + 3u32],
        free_b,
        props[pa],
        props[pa + 1u32],
        0.0 - wind_x,
        roll_ab,
    ) == 1u32
    {
        let cell = b;
        b = a;
        a = cell;
        let row = pb;
        pb = pa;
        pa = row;
        let fall = free_b;
        free_b = free_a;
        free_a = fall;
    }

    // Bottom row.
    if slides(
        props[pc],
        props[pc + 1u32],
        props[pc + 3u32],
        free_c,
        props[pd],
        props[pd + 1u32],
        wind_x,
        roll_cd,
    ) == 1u32
    {
        let cell = c;
        c = d;
        d = cell;
        let row = pc;
        pc = pd;
        pd = row;
        let fall = free_c;
        free_c = free_d;
        free_d = fall;
    } else if slides(
        props[pd],
        props[pd + 1u32],
        props[pd + 3u32],
        free_d,
        props[pc],
        props[pc + 1u32],
        0.0 - wind_x,
        roll_cd,
    ) == 1u32
    {
        let cell = d;
        d = c;
        c = cell;
        let row = pd;
        pd = pc;
        pc = row;
        let fall = free_d;
        free_d = free_c;
        free_c = fall;
    }

    // Up and down the columns, for a gust swept that way. Liquids do not spread
    // vertically, so the runniness is passed as zero and only the wind can fire.
    // The same roll decided whether the column was held up, so a grain the
    // updraft is holding is the one it then lifts, and it rises steadily rather
    // than bobbing. A cell on the ground is kicked up by the sideways wind as
    // well; one already flying only rises in a true updraft.
    let mut rise_c = lift;
    if free_c == 0u32 {
        rise_c = hold;
    }
    let mut rise_d = lift;
    if free_d == 0u32 {
        rise_d = hold;
    }
    if slides(
        props[pa],
        props[pa + 1u32],
        0u32,
        free_a,
        props[pc],
        props[pc + 1u32],
        wind_y,
        roll_ac,
    ) == 1u32
    {
        let cell = a;
        a = c;
        c = cell;
        let row = pa;
        pa = pc;
        pc = row;
        let fall = free_a;
        free_a = free_c;
        free_c = fall;
    } else if slides(
        props[pc],
        props[pc + 1u32],
        0u32,
        free_c,
        props[pa],
        props[pa + 1u32],
        rise_c,
        roll_ac,
    ) == 1u32
    {
        let cell = c;
        c = a;
        a = cell;
        let row = pc;
        pc = pa;
        pa = row;
        let fall = free_c;
        free_c = free_a;
        free_a = fall;
    }
    if slides(
        props[pb],
        props[pb + 1u32],
        0u32,
        free_b,
        props[pd],
        props[pd + 1u32],
        wind_y,
        roll_bd,
    ) == 1u32
    {
        let cell = b;
        b = d;
        d = cell;
        let row = pb;
        pb = pd;
        pd = row;
        let fall = free_b;
        free_b = free_d;
        free_d = fall;
    } else if slides(
        props[pd],
        props[pd + 1u32],
        0u32,
        free_d,
        props[pb],
        props[pb + 1u32],
        rise_d,
        roll_bd,
    ) == 1u32
    {
        let cell = d;
        d = b;
        b = cell;
        let row = pd;
        pd = pb;
        pb = row;
        let fall = free_d;
        free_d = free_b;
        free_b = fall;
    }

    // Put the block back. Every cell of the world belongs to exactly one block
    // on any given pass, so between them these writes fill the whole buffer.
    if left_in == 1u32 && top_in == 1u32 {
        dst[index_a] = a;
    }
    if right_in == 1u32 && top_in == 1u32 {
        dst[index_b] = b;
    }
    if left_in == 1u32 && low_in == 1u32 {
        dst[index_c] = c;
    }
    if right_in == 1u32 && low_in == 1u32 {
        dst[index_d] = d;
    }
}

/// Stamp a filled circle of one material into the grid: the brush.
///
/// Painting material zero, which is air, is how the eraser works.
///
/// `world` is the width, the height and a seed that changes every stroke, so two
/// strokes in the same place do not come out with the same grain. `brush` is the
/// centre, the radius and the material.
#[kernel(workgroup_size(8, 8))]
pub fn paint(cells: &mut [u32], world: &Vec4<u32>, brush: &Vec4<i32>) {
    /// A cheap integer hash, the same one the simulation kernels use.
    fn hash(seed: u32) -> u32 {
        let mut v = seed;
        v ^= v >> 16u32;
        v *= 2246822519u32;
        v ^= v >> 13u32;
        v *= 3266489917u32;
        v ^= v >> 16u32;
        v
    }

    let width = world.x;
    let height = world.y;
    let seed = world.z;
    let radius = brush.z;

    // The dispatch covers the brush's bounding square, so the invocation id is
    // an offset from the centre rather than a place in the world.
    let dx = global_id().x as i32 - radius;
    let dy = global_id().y as i32 - radius;
    if dx * dx + dy * dy > radius * radius {
        return;
    }
    let x = brush.x + dx;
    let y = brush.y + dy;
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }

    let index = y as u32 * width + x as u32;
    // Frozen at the moment it is painted, so a grain keeps its shade as it moves
    // rather than shimmering.
    let variant = hash(seed + index * 2654435761u32) & 255u32;
    cells[index] = brush.w as u32 | (variant << 8u32);
}

/// Blow a gust into a filled circle of the wind field: the wind tool.
///
/// `push` is the velocity to add, in cells per tick. It is strongest at the
/// centre and fades to nothing at the rim, so a sweep leaves a smooth puff of
/// moving air rather than a hard-edged disc of it, and the result is clamped to
/// the top speed in `weather` (see [`weather`]) so holding the tool still does
/// not wind the field up for ever.
#[kernel(workgroup_size(8, 8))]
pub fn gust(
    wind: &mut [Vec2<f32>],
    world: &Vec4<u32>,
    brush: &Vec4<i32>,
    push: &Vec4<f32>,
    weather: &Vec4<f32>,
) {
    let width = world.x;
    let height = world.y;
    let radius = brush.z;
    let top = weather.x;

    let dx = global_id().x as i32 - radius;
    let dy = global_id().y as i32 - radius;
    let distance = dx * dx + dy * dy;
    if distance > radius * radius {
        return;
    }
    let x = brush.x + dx;
    let y = brush.y + dy;
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }

    let falloff = 1.0 - distance as f32 / (radius * radius + 1) as f32;
    let index = y as u32 * width + x as u32;
    let here = wind[index];
    let mut vx = here.x + push.x * falloff;
    let mut vy = here.y + push.y * falloff;
    let speed = sqrt(vx * vx + vy * vy);
    if speed > top {
        vx = vx * top / speed;
        vy = vy * top / speed;
    }
    wind[index] = vec2(vx, vy);
}

/// Carry the wind along by itself, and let it fade.
///
/// Every cell asks where the air now in it came from, one tick ago, and takes
/// the velocity that was there. Looking backwards like that, rather than pushing
/// each cell's velocity forwards, means the answer is always defined and never
/// blows up however fast the wind is. The place it lands between cells is read
/// by blending the four around it. What comes out is scaled by the decay in
/// `weather` (see [`weather`]), so a gust dies away on its own.
///
/// A cell that is solid has no wind in it, and one that is sand or water loses
/// a fifth of it a tick. Damping it here is what makes the pressure solve
/// see a wall as something the air has to go round, and a heap as something it
/// mostly goes over, which is what gives the windward face of a dune its
/// updraft. The result is also held under the top speed in `weather`, since the
/// swirl can otherwise wind an eddy up without limit.
#[kernel(workgroup_size(16, 16))]
pub fn flow(
    src: &[Vec2<f32>],
    dst: &mut [Vec2<f32>],
    cells: &[u32],
    props: &[u32],
    world: &Vec4<u32>,
    weather: &Vec4<f32>,
) {
    let width = world.x;
    let height = world.y;
    let top = weather.x;
    let decay = weather.y;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let material = cells[index] & 255u32;
    let flags = props[material * 8u32 + 1u32];
    if (flags & 2u32) == 0u32 {
        dst[index] = vec2(0.0, 0.0);
        return;
    }
    let mut porosity = 1.0;
    if (flags & 1u32) != 0u32 {
        porosity = 0.8;
    }

    // Where this cell's air was a tick ago, in cell units from the corner of
    // the world, then in index units where the centre of cell `i` sits at `i`.
    let here = src[index];
    let sx = clamp(x as f32 - here.x, 0.0, width as f32 - 1.0);
    let sy = clamp(y as f32 - here.y, 0.0, height as f32 - 1.0);
    let fx = floor(sx);
    let fy = floor(sy);
    let tx = sx - fx;
    let ty = sy - fy;
    let x0 = fx as u32;
    let y0 = fy as u32;
    let x1 = min(x0 + 1u32, width - 1u32);
    let y1 = min(y0 + 1u32, height - 1u32);

    let v00 = src[y0 * width + x0];
    let v10 = src[y0 * width + x1];
    let v01 = src[y1 * width + x0];
    let v11 = src[y1 * width + x1];
    let top_x = v00.x + (v10.x - v00.x) * tx;
    let top_y = v00.y + (v10.y - v00.y) * tx;
    let low_x = v01.x + (v11.x - v01.x) * tx;
    let low_y = v01.y + (v11.y - v01.y) * tx;
    let scale = decay * porosity;
    let mut vx = (top_x + (low_x - top_x) * ty) * scale;
    let mut vy = (top_y + (low_y - top_y) * ty) * scale;
    let speed = sqrt(vx * vx + vy * vy);
    if speed > top {
        vx = vx * top / speed;
        vy = vy * top / speed;
    }

    dst[index] = vec2(vx, vy);
}

/// Measure how much air each cell is gaining or losing: the divergence of the
/// wind. Positive means more is flowing out than in.
///
/// Beyond the edge of the world the air stands still, so an edge cell sees zero
/// on that side.
#[kernel(workgroup_size(16, 16))]
pub fn divergence(wind: &[Vec2<f32>], div: &mut [f32], world: &Vec4<u32>) {
    let width = world.x;
    let height = world.y;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let mut left = 0.0;
    if x > 0u32 {
        left = wind[index - 1u32].x;
    }
    let mut right = 0.0;
    if x + 1u32 < width {
        right = wind[index + 1u32].x;
    }
    let mut up = 0.0;
    if y > 0u32 {
        up = wind[index - width].y;
    }
    let mut down = 0.0;
    if y + 1u32 < height {
        down = wind[index + width].y;
    }
    div[index] = 0.5 * (right - left + down - up);
}

/// One relaxation step towards the pressure that would make the wind
/// incompressible: each cell takes the average of its neighbours, less the
/// divergence measured there.
///
/// This is Jacobi iteration, the simplest solver there is. It converges slowly,
/// but the host runs it [`PRESSURE_ITERATIONS`] times a tick starting from the
/// last tick's answer, and the wind changes little between ticks, so it keeps
/// up. Beyond the edge of the world the pressure is taken to match the edge
/// cell, which is what stops air from being pushed out through it.
///
/// `carry` scales what is read, which is the same as scaling the guess the
/// step starts from. The host passes [`PRESSURE_CARRY`] for the first step of a
/// tick and one for the rest, so the previous tick's answer is leant on but
/// not for ever.
#[kernel(workgroup_size(16, 16))]
pub fn pressure(src: &[f32], dst: &mut [f32], div: &[f32], world: &Vec4<u32>, carry: &f32) {
    let width = world.x;
    let height = world.y;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let here = src[index];
    let mut left = here;
    if x > 0u32 {
        left = src[index - 1u32];
    }
    let mut right = here;
    if x + 1u32 < width {
        right = src[index + 1u32];
    }
    let mut up = here;
    if y > 0u32 {
        up = src[index - width];
    }
    let mut down = here;
    if y + 1u32 < height {
        down = src[index + width];
    }
    dst[index] = ((left + right + up + down) * carry - div[index]) * 0.25;
}

/// Take the pressure gradient out of the wind, which leaves it incompressible:
/// air that was piling up somewhere now flows round instead. This is the step
/// that turns a stamped puff into a gust with eddies at its edges.
///
/// Solid cells are zeroed again here, and sand and water damped again, since
/// the gradient can point into one. The result goes to a fresh buffer so that
/// the live wind is always in the same place, the way the live grid is.
#[kernel(workgroup_size(16, 16))]
pub fn project(
    src: &[Vec2<f32>],
    dst: &mut [Vec2<f32>],
    pressure: &[f32],
    cells: &[u32],
    props: &[u32],
    world: &Vec4<u32>,
) {
    let width = world.x;
    let height = world.y;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let material = cells[index] & 255u32;
    let flags = props[material * 8u32 + 1u32];
    if (flags & 2u32) == 0u32 {
        dst[index] = vec2(0.0, 0.0);
        return;
    }
    let mut porosity = 1.0;
    if (flags & 1u32) != 0u32 {
        porosity = 0.8;
    }

    let here = pressure[index];
    let mut left = here;
    if x > 0u32 {
        left = pressure[index - 1u32];
    }
    let mut right = here;
    if x + 1u32 < width {
        right = pressure[index + 1u32];
    }
    let mut up = here;
    if y > 0u32 {
        up = pressure[index - width];
    }
    let mut down = here;
    if y + 1u32 < height {
        down = pressure[index + width];
    }
    let v = src[index];
    dst[index] = vec2(
        (v.x - 0.5 * (right - left)) * porosity,
        (v.y - 0.5 * (down - up)) * porosity,
    );
}

/// Measure how fast the air is spinning at each cell: the curl of the wind.
/// Positive is clockwise on screen.
#[kernel(workgroup_size(16, 16))]
pub fn curl(wind: &[Vec2<f32>], curl: &mut [f32], world: &Vec4<u32>) {
    let width = world.x;
    let height = world.y;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let mut left = 0.0;
    if x > 0u32 {
        left = wind[index - 1u32].y;
    }
    let mut right = 0.0;
    if x + 1u32 < width {
        right = wind[index + 1u32].y;
    }
    let mut up = 0.0;
    if y > 0u32 {
        up = wind[index - width].x;
    }
    let mut down = 0.0;
    if y + 1u32 < height {
        down = wind[index + width].x;
    }
    curl[index] = 0.5 * ((right - left) - (down - up));
}

/// Wind the eddies up: vorticity confinement.
///
/// Carrying the wind along on a grid smears it, and what it smears away first
/// is the spin, so a gust that ought to curl over into eddies just fades into a
/// smooth drift instead. This puts the spin back. Each cell looks for where the
/// air nearby is spinning hardest and pushes its own wind round that centre, by
/// the swirl strength in `weather` (see [`weather`]) times how hard it is
/// spinning itself. It is the standard trick, and the one sandspiel's fluid
/// leans on for its look.
#[kernel(workgroup_size(16, 16))]
pub fn swirl(curl: &[f32], wind: &mut [Vec2<f32>], world: &Vec4<u32>, weather: &Vec4<f32>) {
    let width = world.x;
    let height = world.y;
    let strength = weather.z;

    let x = global_id().x;
    let y = global_id().y;
    if x >= width {
        return;
    }
    if y >= height {
        return;
    }
    let index = y * width + x;

    let here = curl[index];
    let mut left = 0.0;
    if x > 0u32 {
        left = abs(curl[index - 1u32]);
    }
    let mut right = 0.0;
    if x + 1u32 < width {
        right = abs(curl[index + 1u32]);
    }
    let mut up = 0.0;
    if y > 0u32 {
        up = abs(curl[index - width]);
    }
    let mut down = 0.0;
    if y + 1u32 < height {
        down = abs(curl[index + width]);
    }
    // Which way the spin gets stronger, as a unit vector, then turned a
    // quarter turn so the push goes round the centre rather than into it.
    let gx = 0.5 * (right - left);
    let gy = 0.5 * (down - up);
    let size = sqrt(gx * gx + gy * gy) + 0.0001;
    let push = strength * here / size;
    let v = wind[index];
    wind[index] = vec2(v.x + push * gy, v.y - push * gx);
}
