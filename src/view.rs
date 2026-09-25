//! Where the window is looking: the zoom, and which part of the world is in
//! view.
//!
//! The world is drawn at its own resolution into an offscreen image, and the
//! composite pass (see `bloom.wgsl`) fits a rectangle of that image to the
//! window. At a zoom of one the rectangle is the whole world, which is how the
//! game opens; zoomed in, it is a smaller piece, the same shape as the whole,
//! and it is kept inside the world, so there is never a margin of nothing
//! showing at an edge. The rectangle is described in fractions of the world's
//! width and height, so it means the same on the desktop grid and a phone's.
//!
//! Zooming happens about a point, the cursor or the middle of a pinch, and
//! that point stays over the same cell as the zoom changes, which is what makes
//! zooming into a spot with the wheel feel right. The same rectangle is what
//! the cursor is mapped through to find the cell under it.

/// The zoom the game opens at: the whole world in the window.
pub const MIN_ZOOM: f32 = 1.0;

/// The most the world can be blown up, as a multiple of fitting the window.
/// On a desktop, where the window shows about a third of a cell per pixel at
/// a zoom of one, this is about ten pixels a cell: plenty to watch a grain
/// tumble, and short of a screen full of a few square blocks.
pub const MAX_ZOOM: f32 = 32.0;

/// The zoom, and the world point at the middle of the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// How many times larger than fitting the window the world is drawn,
    /// between [`MIN_ZOOM`] and [`MAX_ZOOM`].
    zoom: f32,
    /// The world point at the middle of the window, as fractions of the
    /// world's width and height, kept so the whole window stays inside the
    /// world.
    center: (f32, f32),
}

impl Default for View {
    /// The whole world, fitted to the window.
    fn default() -> Self {
        Self {
            zoom: MIN_ZOOM,
            center: (0.5, 0.5),
        }
    }
}

impl View {
    /// How much of the world's width and height the window shows.
    fn size(&self) -> (f32, f32) {
        (1.0 / self.zoom, 1.0 / self.zoom)
    }

    /// Bring the zoom into range and the window back inside the world.
    fn settle(&mut self) {
        self.zoom = self.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        let (w, h) = self.size();
        self.center.0 = self.center.0.clamp(w / 2.0, 1.0 - w / 2.0);
        self.center.1 = self.center.1.clamp(h / 2.0, 1.0 - h / 2.0);
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Whether the whole world is in view.
    pub fn is_whole(&self) -> bool {
        self.zoom <= MIN_ZOOM
    }

    /// The rectangle of the world in view, as its top left corner and its
    /// size, in fractions of the world's width and height.
    pub fn visible(&self) -> ((f32, f32), (f32, f32)) {
        let (w, h) = self.size();
        ((self.center.0 - w / 2.0, self.center.1 - h / 2.0), (w, h))
    }

    /// The same rectangle packed for the composite shader: the corner, then
    /// the size.
    pub fn uniform(&self) -> [f32; 4] {
        let ((x, y), (w, h)) = self.visible();
        [x, y, w, h]
    }

    /// The world point under a window point, both as fractions of their
    /// width and height.
    pub fn world_at(&self, at: (f32, f32)) -> (f32, f32) {
        let ((x, y), (w, h)) = self.visible();
        (x + at.0 * w, y + at.1 * h)
    }

    /// The cell under a window point given in pixels, for a window and a
    /// grid of the given sizes in pixels and cells.
    pub fn cell(&self, pos: (f64, f64), window: (u32, u32), grid: (u32, u32)) -> (i32, i32) {
        let at = (
            (pos.0 / window.0.max(1) as f64) as f32,
            (pos.1 / window.1.max(1) as f64) as f32,
        );
        let (x, y) = self.world_at(at);
        ((x * grid.0 as f32) as i32, (y * grid.1 as f32) as i32)
    }

    /// Set the zoom outright, keeping the same point in the middle of the
    /// window as far as the edges of the world allow. What the panel's
    /// slider does.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;
        self.settle();
    }

    /// Multiply the zoom by `factor`, about the window point `at`, given as
    /// fractions of the window's width and height. Whatever is under that
    /// point stays under it, unless the world's edge is reached.
    pub fn zoom_by(&mut self, factor: f32, at: (f32, f32)) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let pinned = self.world_at(at);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let (w, h) = self.size();
        // The corner that puts the pinned point back under `at`, and from it
        // the middle of the window.
        let corner = (pinned.0 - at.0 * w, pinned.1 - at.1 * h);
        self.center = (corner.0 + w / 2.0, corner.1 + h / 2.0);
        self.settle();
    }

    /// Move the view by a fraction of the window's width and height: a
    /// positive `dx` looks further right.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (w, h) = self.size();
        self.center.0 += dx * w;
        self.center.1 += dy * h;
        self.settle();
    }

    /// Back to the whole world in the window.
    pub fn reset(&mut self) {
        *self = View::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    #[test]
    fn opens_on_the_whole_world() {
        let view = View::default();
        assert!(view.is_whole());
        assert_eq!(view.uniform(), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            view.cell((550.0, 155.0), (1100, 620), (3000, 1500)),
            (1500, 375)
        );
    }

    #[test]
    fn zooming_keeps_the_point_under_the_cursor_still() {
        let mut view = View::default();
        let at = (0.3, 0.7);
        let before = view.world_at(at);
        view.zoom_by(4.0, at);
        assert_eq!(view.zoom(), 4.0);
        assert!(close(view.world_at(at), before));
        assert!(!view.is_whole());
        // And a quarter of the world is showing.
        let (_, size) = view.visible();
        assert!(close(size, (0.25, 0.25)));
    }

    #[test]
    fn the_window_never_leaves_the_world() {
        let mut view = View::default();
        // Zooming in at a corner cannot keep the corner in place without
        // showing past the edge, so the view stays flush with it.
        view.zoom_by(2.0, (1.0, 1.0));
        let ((x, y), (w, h)) = view.visible();
        assert!(close((x + w, y + h), (1.0, 1.0)));
        // Panning off the other way is stopped at the far edge.
        view.pan(-10.0, -10.0);
        assert!(close(view.visible().0, (0.0, 0.0)));
        // And zooming out again is the whole world, exactly.
        view.zoom_by(0.01, (0.2, 0.9));
        assert_eq!(view.uniform(), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn the_zoom_stays_within_bounds() {
        let mut view = View::default();
        view.zoom_by(1000.0, (0.5, 0.5));
        assert_eq!(view.zoom(), MAX_ZOOM);
        view.set_zoom(0.0);
        assert_eq!(view.zoom(), MIN_ZOOM);
        view.zoom_by(f32::NAN, (0.5, 0.5));
        assert_eq!(
            view.zoom(),
            MIN_ZOOM,
            "a pinch that reports nothing does nothing"
        );
        view.zoom_by(-2.0, (0.5, 0.5));
        assert_eq!(view.zoom(), MIN_ZOOM);
    }

    #[test]
    fn panning_moves_by_a_fraction_of_the_window() {
        let mut view = View::default();
        view.set_zoom(4.0);
        assert!(
            close(view.visible().0, (0.375, 0.375)),
            "zoomed about the middle"
        );
        view.pan(0.5, 0.0);
        // Half a window at a quarter of the world is an eighth of the world.
        assert!(close(view.visible().0, (0.5, 0.375)));
        view.reset();
        assert!(view.is_whole());
    }

    #[test]
    fn the_cursor_maps_through_the_view() {
        let mut view = View::default();
        view.set_zoom(2.0);
        // Half the world is showing, centred, so the window's corner is a
        // quarter of the way in.
        assert_eq!(view.cell((0.0, 0.0), (1100, 620), (3000, 1500)), (750, 375));
        assert_eq!(
            view.cell((1100.0, 620.0), (1100, 620), (3000, 1500)),
            (2250, 1125)
        );
    }
}
