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
//!
//! A [`Camera`] is what the game actually holds: the view the input is
//! heading for, and the one on screen, which eases after it a little every
//! frame. A notch of the wheel, a key or the slider sets where to go and the
//! window glides there. A drag or a pinch moves the picture under the fingers
//! directly, since a view that trailed behind them would feel loose.

/// The zoom the game opens at: the whole world in the window.
pub const MIN_ZOOM: f32 = 1.0;

/// The most the world can be blown up, as a multiple of fitting the window.
/// On a desktop, where the window shows about a third of a cell per pixel at
/// a zoom of one, this is about ten pixels a cell: plenty to watch a grain
/// tumble, and short of a screen full of a few square blocks.
pub const MAX_ZOOM: f32 = 32.0;

/// How quickly the view on screen catches up with where it is heading: the
/// gap shrinks by a factor of e this many times a second, so it is nearly
/// closed in a sixth of a second whatever the frame rate.
const EASE_RATE: f32 = 18.0;

/// A gap smaller than this, in fractions of the world, is closed outright
/// rather than eased forever.
const SNAP: f32 = 1e-5;

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
        self.pin(self.zoom * factor, self.world_at(at), at);
    }

    /// Set the zoom and put the world point `pinned` under the window point
    /// `at`, as far as the edges of the world allow.
    fn pin(&mut self, zoom: f32, pinned: (f32, f32), at: (f32, f32)) {
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        let (w, h) = self.size();
        // The corner that puts the pinned point under `at`, and from it the
        // middle of the window.
        let corner = (pinned.0 - at.0 * w, pinned.1 - at.1 * h);
        self.center = (corner.0 + w / 2.0, corner.1 + h / 2.0);
        self.settle();
    }

    /// Move the view by a fraction of the window's width and height: a
    /// positive `dx` looks further right.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (w, h) = self.size();
        self.shift(dx * w, dy * h);
    }

    /// Move the view by a distance in fractions of the world.
    fn shift(&mut self, dx: f32, dy: f32) {
        self.center.0 += dx;
        self.center.1 += dy;
        self.settle();
    }

    /// Step part of the way from here towards `to`, `t` being the part,
    /// from nothing to all of it. The size of the window and its middle move
    /// in straight lines, which keeps a point pinned under the cursor by a
    /// zoom in the same place all the way there, and never leaves the world,
    /// since both ends are inside it.
    fn ease_towards(&mut self, to: &View, t: f32) {
        let (from_size, to_size) = (1.0 / self.zoom, 1.0 / to.zoom);
        let size = from_size + (to_size - from_size) * t;
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        self.zoom = 1.0 / size;
        self.center = (
            lerp(self.center.0, to.center.0),
            lerp(self.center.1, to.center.1),
        );
        let gap = (size - to_size)
            .abs()
            .max((self.center.0 - to.center.0).abs())
            .max((self.center.1 - to.center.1).abs());
        if gap < SNAP {
            *self = *to;
        }
    }

    /// Back to the whole world in the window.
    pub fn reset(&mut self) {
        *self = View::default();
    }
}

/// The view on screen, and the one it is easing towards. See the top of this
/// module.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Camera {
    /// Where the view is heading. The slider shows this zoom, so it does not
    /// wobble as the picture catches up.
    target: View,
    /// What the window shows this frame, and what the cursor is mapped
    /// through.
    shown: View,
}

impl Camera {
    /// What the window shows this frame.
    pub fn shown(&self) -> &View {
        &self.shown
    }

    /// The zoom the view is heading for.
    pub fn zoom(&self) -> f32 {
        self.target.zoom()
    }

    /// Whether the view is at, or on its way back to, the whole world.
    pub fn is_whole(&self) -> bool {
        self.target.is_whole()
    }

    /// Ease the view on screen towards where it is heading, `dt` seconds
    /// after the last frame.
    pub fn tick(&mut self, dt: f32) {
        let t = 1.0 - (-EASE_RATE * dt.max(0.0)).exp();
        self.shown.ease_towards(&self.target, t);
    }

    /// Glide to a zoom, about the middle of the window.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.target.set_zoom(zoom);
    }

    /// Glide in or out by `factor` about the window point `at`, as a notch
    /// of the wheel or a key does. It is the point under `at` on screen now
    /// that is kept there, so a notch turned while the last is still playing
    /// out zooms into what the cursor is actually over.
    pub fn zoom_by(&mut self, factor: f32, at: (f32, f32)) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let pinned = self.shown.world_at(at);
        self.target.pin(self.target.zoom * factor, pinned, at);
    }

    /// Zoom by `factor` about `at` straight away, as a pinch does, so the
    /// picture stays under the fingers. Anything still playing out stops.
    pub fn pinch_by(&mut self, factor: f32, at: (f32, f32)) {
        self.shown.zoom_by(factor, at);
        self.target = self.shown;
    }

    /// Glide by a fraction of the window's width and height, as the keys do.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.target.pan(dx, dy);
    }

    /// Move by a fraction of the window straight away, as a drag does, so
    /// the world stays under the cursor. A zoom still playing out carries on
    /// from the new place.
    pub fn drag(&mut self, dx: f32, dy: f32) {
        let (w, h) = self.shown.size();
        let (dx, dy) = (dx * w, dy * h);
        self.shown.shift(dx, dy);
        self.target.shift(dx, dy);
    }

    /// Glide back to the whole world.
    pub fn reset(&mut self) {
        self.target.reset();
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
    fn the_camera_glides_to_a_zoom_and_keeps_the_cursor_still_on_the_way() {
        let mut camera = Camera::default();
        let at = (0.3, 0.7);
        let before = camera.shown().world_at(at);
        camera.zoom_by(4.0, at);
        // Nothing has moved yet, but the slider already shows where it is going.
        assert!(camera.shown().is_whole());
        assert_eq!(camera.zoom(), 4.0);
        camera.tick(1.0 / 60.0);
        let zoom = camera.shown().zoom();
        assert!(zoom > 1.0 && zoom < 4.0, "part of the way in: {zoom}");
        assert!(close(camera.shown().world_at(at), before));
        // And a second or so later it is there, exactly.
        for _ in 0..60 {
            camera.tick(1.0 / 60.0);
        }
        assert_eq!(camera.shown().zoom(), 4.0);
        assert!(close(camera.shown().world_at(at), before));
    }

    #[test]
    fn a_second_notch_zooms_into_what_is_on_screen() {
        let mut camera = Camera::default();
        camera.zoom_by(2.0, (0.5, 0.5));
        camera.tick(1.0 / 60.0);
        // Halfway through, the cursor moves and the wheel turns again.
        let at = (0.8, 0.2);
        let under = camera.shown().world_at(at);
        camera.zoom_by(2.0, at);
        camera.tick(10.0);
        assert_eq!(camera.shown().zoom(), 4.0);
        assert!(close(camera.shown().world_at(at), under));
    }

    #[test]
    fn a_drag_moves_the_picture_at_once() {
        let mut camera = Camera::default();
        camera.pinch_by(4.0, (0.5, 0.5));
        assert_eq!(camera.shown().zoom(), 4.0, "a pinch does not trail");
        camera.drag(0.5, 0.0);
        assert!(close(camera.shown().visible().0, (0.5, 0.375)));
        camera.tick(10.0);
        assert!(close(camera.shown().visible().0, (0.5, 0.375)));
    }

    #[test]
    fn the_camera_glides_back_to_the_whole_world() {
        let mut camera = Camera::default();
        camera.pinch_by(8.0, (0.1, 0.9));
        camera.reset();
        assert!(camera.is_whole());
        assert!(!camera.shown().is_whole());
        camera.tick(10.0);
        assert_eq!(camera.shown().uniform(), [0.0, 0.0, 1.0, 1.0]);
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
