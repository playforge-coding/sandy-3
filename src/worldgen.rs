//! World generation: the pieces a world preset script builds a landscape out
//! of.
//!
//! A world preset is a plugin (see [`crate::plugins`]) that registers a name
//! and a `generate` function. When the panel asks for that world, the function
//! is handed a [`Canvas`], a grid of materials in main memory the same size as
//! the world, and paints into it with `set`, `fill` and `disk`. Once the
//! function returns, the finished grid is uploaded to the GPU in one go, which
//! is why there is a canvas at all: five hundred thousand cells are far better
//! sent as one buffer write than as five hundred thousand brush strokes.
//!
//! The shape of the land comes from [`Noise`], a wrapper over
//! [FastNoise2](https://github.com/Auburn/FastNoise2) built from a small table
//! of knobs: the kind of noise, its frequency, and how many octaves to stack.
//! A script samples it a cell at a time, or asks for a whole grid of it, which
//! FastNoise2 fills with SIMD in a few milliseconds.
//!
//! Both types are Lua userdata, so a script sees them as objects with
//! methods. Everything else about generation, which script is which world and
//! how the result reaches the GPU, is in [`crate::plugins`] and
//! [`crate::sim::Simulation::load`].

use fastnoise2::SafeNode;
use fastnoise2::generator::perlin::Perlin;
use fastnoise2::generator::prelude::*;
use fastnoise2::generator::simplex::{Simplex, SuperSimplex};
use fastnoise2::generator::value::Value as ValueNoise;
use mlua::{Table, UserData, UserDataMethods};

use crate::materials::MaterialId;
use crate::plugins::{optional, runtime};

/// The most cells of noise a script can ask for at once. A grid the size of
/// the world is half a million floats, which is fine; this stops a typo asking
/// for a billion.
const MAX_NOISE_CELLS: usize = 1 << 24;

/// A world being built: one material per cell, in main memory. Coordinates
/// are grid cells with the origin at the top left, as everywhere else in the
/// game. Every method clips to the grid, so a script can paint a tree that
/// pokes off the top of the world without checking first.
pub struct Canvas {
    width: u32,
    height: u32,
    cells: Vec<MaterialId>,
}

impl Canvas {
    /// An empty world, all air.
    pub fn new(width: u32, height: u32) -> Self {
        Canvas {
            width,
            height,
            cells: vec![0; (width * height) as usize],
        }
    }

    fn index(&self, x: i64, y: i64) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            None
        } else {
            Some((y as u32 * self.width + x as u32) as usize)
        }
    }

    /// The material at a cell, or `None` outside the grid.
    pub fn get(&self, x: i64, y: i64) -> Option<MaterialId> {
        self.index(x, y).map(|i| self.cells[i])
    }

    /// Put a material in one cell. Outside the grid, nothing happens.
    pub fn set(&mut self, x: i64, y: i64, material: MaterialId) {
        if let Some(i) = self.index(x, y) {
            self.cells[i] = material;
        }
    }

    /// Fill a rectangle, both corners included, given in either order.
    pub fn fill(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, material: MaterialId) {
        let (x0, x1) = (x0.min(x1).max(0), x0.max(x1).min(self.width as i64 - 1));
        let (y0, y1) = (y0.min(y1).max(0), y0.max(y1).min(self.height as i64 - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = (y as u32 * self.width + x as u32) as usize;
                self.cells[i] = material;
            }
        }
    }

    /// Fill a circle, the way the plain brush does.
    pub fn disk(&mut self, cx: i64, cy: i64, radius: i64, material: MaterialId) {
        let radius = radius.max(0);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    self.set(cx + dx, cy + dy, material);
                }
            }
        }
    }

    /// The finished grid, row by row from the top, for
    /// [`crate::sim::Simulation::load`].
    pub fn into_cells(self) -> Vec<MaterialId> {
        self.cells
    }
}

/// A noise field a script can sample, built from `sandy.noise { ... }`.
///
/// The value at a point is FastNoise2's, roughly in `-1..=1`, sampled at the
/// cell coordinates. The frequency goes into the generator itself, as
/// FastNoise2's feature scale (its reciprocal), so a frequency of `0.01`
/// means a feature every hundred cells or so. The seed is fixed when the
/// noise is made, so the same spec and seed give the same field every time.
pub struct Noise {
    node: SafeNode,
    seed: i32,
    /// What the raw output is multiplied by. FastNoise2 adds its octaves up
    /// as they are, so four of them at a gain of a half reach nearly twice
    /// as far as one; this brings the sum back to about `-1..=1`, the way
    /// FastNoise Lite bounded its fractals, so a script can size hills
    /// without caring how many octaves are under them.
    scale: f32,
}

impl Noise {
    /// Build a field from a script's spec table. The fields are `seed`
    /// (default 0), `kind` (`simplex`, the default, `supersimplex`, `perlin`
    /// or `value`), `frequency` (default 0.01), `octaves` (default 1, which is
    /// plain noise; more stacks finer detail on top), `gain` and `lacunarity`
    /// (how much quieter and how much finer each octave is; 0.5 and 2), and
    /// `ridged`, which makes the octaves fold into ridges instead of hills.
    pub fn from_spec(spec: &Table) -> mlua::Result<Self> {
        let seed: i64 = optional(spec, "seed", 0)?;
        let kind: String = optional(spec, "kind", "simplex".to_string())?;
        let frequency: f64 = optional(spec, "frequency", 0.01)?;
        let octaves: i64 = optional(spec, "octaves", 1)?;
        let gain: f64 = optional(spec, "gain", 0.5)?;
        let lacunarity: f64 = optional(spec, "lacunarity", 2.0)?;
        let ridged: bool = optional(spec, "ridged", false)?;

        if !(frequency.is_finite() && frequency > 0.0) {
            return Err(runtime("frequency should be a positive number"));
        }
        if !(1..=16).contains(&octaves) {
            return Err(runtime("octaves should be from 1 to 16"));
        }

        // FastNoise2 sizes its features in world units rather than taking a
        // frequency: a feature scale of a hundred is a feature every hundred
        // cells. The generators default to that, which is why the scale is
        // set here rather than left alone.
        let feature_scale = (1.0 / frequency) as f32;
        let base: GeneratorWrapper<SafeNode> = match kind.to_ascii_lowercase().as_str() {
            "simplex" => Simplex {
                feature_scale,
                ..Default::default()
            }
            .build(),
            "supersimplex" => SuperSimplex {
                feature_scale,
                ..Default::default()
            }
            .build(),
            "perlin" => Perlin {
                feature_scale,
                ..Default::default()
            }
            .build(),
            "value" => ValueNoise {
                feature_scale,
                ..Default::default()
            }
            .build(),
            other => {
                return Err(runtime(format!(
                    "kind should be simplex, supersimplex, perlin or value, not '{other}'"
                )));
            }
        };
        let node = if octaves == 1 {
            base
        } else if ridged {
            base.ridged(gain as f32, 0.0, octaves as i32, lacunarity as f32)
                .build()
        } else {
            base.fbm(gain as f32, 0.0, octaves as i32, lacunarity as f32)
                .build()
        };

        // The most the octaves can add up to: one, plus the gain, plus the
        // gain squared, and so on.
        let bound: f64 = (0..octaves).map(|i| gain.abs().powi(i as i32)).sum();
        Ok(Noise {
            node: node.0,
            // A seed is a whole 32 bits to FastNoise2 and a plain integer to
            // a script, so a seed past the signed range wraps rather than
            // being refused; every number still names a world.
            seed: seed as i32,
            scale: if bound > 0.0 { 1.0 / bound as f32 } else { 1.0 },
        })
    }

    /// The value at one cell.
    pub fn at(&self, x: f32, y: f32) -> f32 {
        self.node.gen_single_2d(x, y, self.seed) * self.scale
    }

    /// The values over a `width` by `height` grid of cells starting at the
    /// origin, row by row from the top, so `grid[y * width + x]` is the same
    /// number [`Noise::at`] gives for `(x, y)`.
    pub fn grid(&self, width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0; width * height];
        if width > 0 && height > 0 {
            self.node.gen_uniform_grid_2d(
                &mut out,
                0.0,
                0.0,
                width as i32,
                height as i32,
                1.0,
                1.0,
                self.seed,
            );
            for v in &mut out {
                *v *= self.scale;
            }
        }
        out
    }
}

impl UserData for Noise {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("at", |_, noise, (x, y): (f64, f64)| {
            Ok(noise.at(x as f32, y as f32) as f64)
        });

        // The grid comes back as a table of rows, `rows[y][x]`, both from
        // zero, so a script reads it with the same coordinates it paints
        // with.
        methods.add_method("grid", |lua, noise, (width, height): (f64, f64)| {
            if !(width >= 1.0 && height >= 1.0) || !(width.is_finite() && height.is_finite()) {
                return Err(runtime("a grid needs a width and a height of at least one"));
            }
            let (width, height) = (width.floor() as usize, height.floor() as usize);
            if width.saturating_mul(height) > MAX_NOISE_CELLS {
                return Err(runtime(format!(
                    "a grid of {width} by {height} is too big; the most is {MAX_NOISE_CELLS} cells"
                )));
            }
            let values = noise.grid(width, height);
            let rows = lua.create_table_with_capacity(height, 0)?;
            for y in 0..height {
                let row = lua.create_table_with_capacity(width, 1)?;
                for x in 0..width {
                    row.raw_set(x, values[y * width + x] as f64)?;
                }
                rows.raw_set(y, row)?;
            }
            Ok(rows)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{EMPTY, SAND, STONE, WATER};
    use mlua::Lua;

    fn spec(lua: &Lua, source: &str) -> Table {
        lua.load(source).eval().unwrap()
    }

    #[test]
    fn the_canvas_clips_everything_to_the_grid() {
        let mut canvas = Canvas::new(10, 5);
        canvas.set(-1, 0, SAND);
        canvas.set(0, -1, SAND);
        canvas.set(10, 0, SAND);
        canvas.set(0, 5, SAND);
        assert_eq!(canvas.get(-1, 0), None);
        assert_eq!(canvas.get(10, 0), None);
        assert!(canvas.into_cells().iter().all(|&m| m == EMPTY));

        let mut canvas = Canvas::new(10, 5);
        // Corners in the wrong order, and hanging off two edges.
        canvas.fill(12, 3, 7, -2, STONE);
        for y in 0..5 {
            for x in 0..10 {
                let expected = if x >= 7 && y <= 3 { STONE } else { EMPTY };
                assert_eq!(canvas.get(x, y), Some(expected), "at ({x}, {y})");
            }
        }

        let mut canvas = Canvas::new(10, 5);
        canvas.disk(0, 0, 2, WATER);
        assert_eq!(canvas.get(0, 0), Some(WATER));
        assert_eq!(canvas.get(2, 0), Some(WATER));
        assert_eq!(canvas.get(2, 2), Some(EMPTY), "outside the circle");
        assert_eq!(
            canvas.into_cells().iter().filter(|&&m| m == WATER).count(),
            6
        );
    }

    #[test]
    fn noise_is_reproducible_and_the_grid_matches_single_samples() {
        let lua = Lua::new();
        let a = Noise::from_spec(&spec(
            &lua,
            "return { seed = 7, frequency = 0.02, octaves = 4 }",
        ))
        .unwrap();
        let b = Noise::from_spec(&spec(
            &lua,
            "return { seed = 7, frequency = 0.02, octaves = 4 }",
        ))
        .unwrap();
        let c = Noise::from_spec(&spec(
            &lua,
            "return { seed = 8, frequency = 0.02, octaves = 4 }",
        ))
        .unwrap();

        let grid = a.grid(50, 20);
        assert_eq!(grid.len(), 1000);
        let mut differs = false;
        for y in 0..20 {
            for x in 0..50 {
                let v = grid[y * 50 + x];
                assert!((-1.5..=1.5).contains(&v), "noise out of range: {v}");
                assert!(
                    (v - a.at(x as f32, y as f32)).abs() < 1e-4,
                    "grid and single sample disagree at ({x}, {y})"
                );
                assert_eq!(v, b.at(x as f32, y as f32), "the same seed differs");
                differs |= v != c.at(x as f32, y as f32);
            }
        }
        assert!(differs, "another seed should give another field");
        let spread = grid.iter().cloned().fold(0.0f32, f32::max)
            - grid.iter().cloned().fold(0.0f32, f32::min);
        assert!(
            spread > 0.5,
            "the field should actually vary, but spans {spread}"
        );
    }

    #[test]
    fn every_kind_of_noise_builds_and_a_bad_spec_says_why() {
        let lua = Lua::new();
        for kind in ["simplex", "SuperSimplex", "perlin", "value"] {
            let noise = Noise::from_spec(&spec(
                &lua,
                &format!("return {{ kind = '{kind}', octaves = 3, ridged = true }}"),
            ))
            .unwrap_or_else(|err| panic!("{kind}: {err}"));
            assert!(noise.at(3.0, 4.0).is_finite());
        }
        let bad = |source: &str| match Noise::from_spec(&spec(&lua, source)) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("{source} should have been refused"),
        };
        assert!(bad("return { kind = 'brown' }").contains("brown"));
        assert!(bad("return { frequency = 0 }").contains("frequency"));
        assert!(bad("return { octaves = 0 }").contains("octaves"));
    }

    #[test]
    fn a_script_reads_the_grid_by_row_and_column_from_zero() {
        let lua = Lua::new();
        let noise = Noise::from_spec(&spec(&lua, "return { seed = 3 }")).unwrap();
        let expected = noise.at(4.0, 2.0) as f64;
        lua.globals().set("n", noise).unwrap();
        let (value, rows, cols): (f64, i64, i64) = lua
            .load(
                r#"
                local g = n:grid(6, 3)
                local rows, cols = 0, 0
                for _ in pairs(g) do rows = rows + 1 end
                for _ in pairs(g[0]) do cols = cols + 1 end
                return g[2][4], rows, cols
                "#,
            )
            .eval()
            .unwrap();
        assert!((value - expected).abs() < 1e-4);
        assert_eq!((rows, cols), (3, 6));
        assert!(
            lua.load("return n:grid(0, 5)").eval::<Table>().is_err(),
            "an empty grid is refused"
        );
    }
}
