//! GIF recordings.
//!
//! A GIF frame has at most 256 colours, and a frame of the world has more:
//! every material comes in a spread of shades, the glow is a gradient and so
//! is the haze. So the colours have to be chosen, and the choosing is what
//! this file is about; the `gif` crate does the rest.
//!
//! The usual ways are one palette for the whole file, which needs every
//! frame before the first can be written, or a fresh palette per frame,
//! which makes a still pile of sand twinkle as each frame picks slightly
//! different shades. A recording here is written as it happens and has to
//! look still where it is still, so it does neither. A [`Palette`] is kept
//! from frame to frame and a colour keeps its entry once it has one. While
//! there are entries free, a new colour simply takes one, so a scene of few
//! colours is exact. Once the palette is full, a new colour is given the
//! nearest entry, and only when a frame brings in a good many colours that
//! have no near entry, which is a new material or a new glow coming into
//! the scene, is the palette built again from that frame with NeuQuant.
//! That costs one visible shift, at the moment the scene itself changed.
//!
//! The frames are deltas. After the first, each is only the rectangle that
//! differs from the frame before, with the pixels inside it that did not
//! change left transparent, so a world that is mostly still costs almost
//! nothing a frame. A rebuilt palette means a whole frame again, so that
//! the screen is never part old palette and part new.

use std::collections::HashMap;
use std::io::{self, Write};

use color_quant::NeuQuant;

use super::{Frame, Rect};

/// The most colours a frame's palette holds. A GIF allows 256, and one is
/// kept back for the transparent entry a delta frame marks its unchanged
/// pixels with.
const PALETTE_SIZE: usize = 255;

/// How far a colour may be from its nearest palette entry and still count
/// as matched: the sum of the squared differences per channel, so this is
/// about ten levels in each.
const TOLERANCE: u32 = 300;

/// The share of a frame's pixels that may be poorly matched before the
/// palette is rebuilt. One in a thousand is a patch a few dozen cells
/// across, so a new material shows true almost as soon as it is painted,
/// while a stray grain or two does not upset the palette.
const REBUILD_SHARE: usize = 1000;

/// NeuQuant looks at one pixel in this many while learning. Ten is the
/// figure its author suggests as the trade between speed and quality.
const SAMPLE_FACTOR: i32 = 10;

/// The shortest a GIF frame can be shown for, in hundredths of a second,
/// and be honoured. Browsers stretch anything shorter out to a tenth of a
/// second, which is the opposite of what was asked for.
const MIN_DELAY: u64 = 2;

/// The colours in use and where each one maps.
struct Palette {
    /// The entries, at most [`PALETTE_SIZE`], as they will be written.
    colors: Vec<[u8; 3]>,
    /// Every colour seen since the palette was last built, to the entry it
    /// was given, and whether that entry was a poor match. The index is the
    /// low byte and the poor-match flag bit eight.
    cache: HashMap<u32, u16>,
}

const POOR: u16 = 0x100;

fn key(px: &[u8]) -> u32 {
    u32::from_be_bytes([0, px[0], px[1], px[2]])
}

fn distance(a: [u8; 3], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).pow(2) as u32)
        .sum()
}

impl Palette {
    fn new() -> Self {
        Palette {
            colors: Vec::with_capacity(PALETTE_SIZE),
            cache: HashMap::new(),
        }
    }

    /// The entry for a colour: its own if it has one, a free one if there
    /// is one, and the nearest otherwise.
    fn entry(&mut self, px: &[u8]) -> u16 {
        let key = key(px);
        if let Some(&entry) = self.cache.get(&key) {
            return entry;
        }
        let entry = if self.colors.len() < PALETTE_SIZE {
            self.colors.push([px[0], px[1], px[2]]);
            (self.colors.len() - 1) as u16
        } else {
            let (index, error) = self
                .colors
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, distance(c, px)))
                .min_by_key(|&(_, error)| error)
                .expect("a full palette is not empty");
            index as u16 | if error > TOLERANCE { POOR } else { 0 }
        };
        self.cache.insert(key, entry);
        entry
    }

    /// Map a frame to indices into the palette, and say whether the palette
    /// served it well enough to use.
    fn map(&mut self, rgb: &[u8], indices: &mut Vec<u8>) -> bool {
        indices.clear();
        let mut poor = 0;
        for px in rgb.as_chunks::<3>().0 {
            let entry = self.entry(px);
            poor += usize::from(entry & POOR != 0);
            indices.push(entry as u8);
        }
        poor <= indices.len() / REBUILD_SHARE
    }

    /// Throw the palette away and choose one for this frame, then map it.
    fn rebuild(&mut self, rgb: &[u8], indices: &mut Vec<u8>) {
        let rgba: Vec<u8> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[1], px[2], 255])
            .collect();
        let quantizer = NeuQuant::new(SAMPLE_FACTOR, PALETTE_SIZE, &rgba);
        self.colors = quantizer.color_map_rgb().as_chunks::<3>().0.to_vec();
        self.cache.clear();
        indices.clear();
        for px in rgba.as_chunks::<4>().0 {
            let key = key(px);
            let entry = match self.cache.get(&key) {
                Some(&entry) => entry,
                None => {
                    // NeuQuant's own search, which is what its palette was
                    // built for. Nothing counts as poor: the palette was
                    // just made for this very frame.
                    let entry = quantizer.index_of(px) as u16;
                    self.cache.insert(key, entry);
                    entry
                }
            };
            indices.push(entry as u8);
        }
    }

    /// The entry after the last colour, which is the one a delta frame uses
    /// for a pixel that has not changed.
    fn transparent(&self) -> u8 {
        self.colors.len() as u8
    }

    /// The palette as the `gif` crate wants it: `r, g, b, r, g, b, ...`,
    /// with the transparent entry on the end so the table reaches it.
    fn flat(&self) -> Vec<u8> {
        let mut flat: Vec<u8> = self.colors.iter().flatten().copied().collect();
        flat.extend_from_slice(&[0, 0, 0]);
        flat
    }
}

/// A GIF, one frame at a time.
pub struct Writer<W: Write> {
    encoder: gif::Encoder<W>,
    width: u16,
    height: u16,
    palette: Palette,
    /// The whole of the latest frame as palette entries.
    indices: Vec<u8>,
    /// The last frame written, to find what the next one changes.
    previous: Option<Vec<u8>>,
    /// Real time the frames so far should have taken, in milliseconds, and
    /// the delays actually written, in hundredths. A GIF delay is a whole
    /// hundredth, so each frame's is chosen to keep the second on the
    /// first's heels rather than rounded on its own, which would drift.
    elapsed_ms: u64,
    written_hundredths: u64,
}

impl<W: Write> Writer<W> {
    pub fn new(w: W, width: u32, height: u32) -> io::Result<Self> {
        let (width, height) = match (u16::try_from(width), u16::try_from(height)) {
            (Ok(w), Ok(h)) => (w, h),
            _ => return Err(io::Error::other("too big for a GIF")),
        };
        // No global palette: each frame carries its own, which is how the
        // palette gets to change when it has to.
        let mut encoder = gif::Encoder::new(w, width, height, &[]).map_err(io::Error::other)?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(io::Error::other)?;
        Ok(Writer {
            encoder,
            width,
            height,
            palette: Palette::new(),
            indices: Vec::new(),
            previous: None,
            elapsed_ms: 0,
            written_hundredths: 0,
        })
    }
}

impl<W: Write> super::AnimationWriter for Writer<W> {
    fn frame(&mut self, frame: &Frame, duration_ms: u32) -> io::Result<()> {
        assert_eq!(
            (frame.width, frame.height),
            (self.width as u32, self.height as u32)
        );
        let rebuilt = !self.palette.map(&frame.rgb, &mut self.indices);
        if rebuilt {
            self.palette.rebuild(&frame.rgb, &mut self.indices);
        }

        self.elapsed_ms += duration_ms as u64;
        let due = self.elapsed_ms.div_ceil(10);
        let delay = due.saturating_sub(self.written_hundredths).max(MIN_DELAY);
        self.written_hundredths += delay;

        // The first frame, and any after a palette rebuild, is the whole
        // picture. Every other one is the rectangle that changed, with only
        // the changed pixels in it drawn. A frame that changed nothing still
        // has to be written to carry its time; one transparent pixel does.
        let width = self.width as u32;
        let transparent = self.palette.transparent();
        let (rect, buffer, transparent) = match (&self.previous, rebuilt) {
            (None, _) | (_, true) => (
                Rect::whole(width, self.height as u32),
                std::borrow::Cow::Borrowed(&self.indices[..]),
                None,
            ),
            (Some(previous), false) => match Rect::changed(previous, &frame.rgb, width) {
                None => (
                    Rect::new(0, 0, 1, 1),
                    std::borrow::Cow::Owned(vec![transparent]),
                    Some(transparent),
                ),
                Some(rect) => {
                    let mut buffer = Vec::with_capacity((rect.width * rect.height) as usize);
                    for (row, (before, after)) in rect.rows(previous, &frame.rgb, width).enumerate()
                    {
                        let start = ((rect.y + row as u32) * width + rect.x) as usize;
                        let indices = &self.indices[start..start + rect.width as usize];
                        let pairs = before
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .zip(after.as_chunks::<3>().0);
                        for ((b, a), &index) in pairs.zip(indices) {
                            buffer.push(if a == b { transparent } else { index });
                        }
                    }
                    (rect, std::borrow::Cow::Owned(buffer), Some(transparent))
                }
            },
        };

        let gif_frame = gif::Frame {
            delay: delay.min(u16::MAX as u64) as u16,
            left: rect.x as u16,
            top: rect.y as u16,
            width: rect.width as u16,
            height: rect.height as u16,
            transparent,
            dispose: gif::DisposalMethod::Keep,
            palette: Some(self.palette.flat()),
            buffer,
            ..Default::default()
        };
        self.encoder
            .write_frame(&gif_frame)
            .map_err(io::Error::other)?;
        match &mut self.previous {
            Some(previous) => previous.copy_from_slice(&frame.rgb),
            None => self.previous = Some(frame.rgb.clone()),
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> io::Result<()> {
        // Writes the trailer and hands the file back.
        let mut w = self.encoder.into_inner().map_err(io::Error::other)?;
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::super::AnimationWriter as _;
    use super::super::tests::frame_of;
    use super::*;
    use std::io::Cursor;

    /// Every frame of a GIF as a viewer would show it, as RGB, with its
    /// delay in hundredths and how many pixels the frame itself carried.
    /// Each frame is drawn over the last where it is opaque.
    fn decode(bytes: Vec<u8>) -> Vec<(Vec<u8>, u16, usize)> {
        let mut options = gif::DecodeOptions::new();
        options.set_color_output(gif::ColorOutput::RGBA);
        let mut decoder = options.read_info(Cursor::new(bytes)).unwrap();
        let width = decoder.width() as usize;
        let mut canvas = vec![0u8; width * decoder.height() as usize * 3];
        let mut frames = Vec::new();
        while let Some(frame) = decoder.read_next_frame().unwrap() {
            for (row, line) in frame
                .buffer
                .chunks_exact(frame.width as usize * 4)
                .enumerate()
            {
                for (col, px) in line.as_chunks::<4>().0.iter().enumerate() {
                    if px[3] != 0 {
                        let at =
                            ((frame.top as usize + row) * width + frame.left as usize + col) * 3;
                        canvas[at..at + 3].copy_from_slice(&px[..3]);
                    }
                }
            }
            frames.push((
                canvas.clone(),
                frame.delay,
                frame.width as usize * frame.height as usize,
            ));
        }
        frames
    }

    #[test]
    fn a_scene_of_few_colours_is_exact_and_keeps_its_time() {
        let a = frame_of(40, 30, &[[0, 0, 0], [255, 200, 20], [30, 60, 200]]);
        let b = frame_of(
            40,
            30,
            &[[255, 200, 20], [0, 0, 0], [90, 90, 90], [30, 60, 200]],
        );
        let mut out = Vec::new();
        let mut writer = Box::new(Writer::new(&mut out, 40, 30).unwrap());
        writer.frame(&a, 33).unwrap();
        writer.frame(&b, 34).unwrap();
        writer.frame(&a, 33).unwrap();
        writer.frame(&b, 5).unwrap();
        writer.finish().unwrap();

        let frames = decode(out);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0].0, a.rgb);
        assert_eq!(frames[1].0, b.rgb);
        assert_eq!(frames[2].0, a.rgb);
        assert_eq!(frames[3].0, b.rgb);
        // 33, 67, 100, 105 ms: 4, 3, 3, then at least the minimum.
        let delays: Vec<u16> = frames.iter().map(|f| f.1).collect();
        assert_eq!(delays, [4, 3, 3, 2]);
    }

    #[test]
    fn frames_after_the_first_carry_only_what_changed() {
        let a = frame_of(40, 30, &[[0, 0, 0], [255, 200, 20], [30, 60, 200]]);
        let mut b = a.clone();
        // One pixel near the top left, one near the bottom right.
        b.rgb[(2 * 40 + 3) * 3..(2 * 40 + 3) * 3 + 3].copy_from_slice(&[9, 9, 9]);
        b.rgb[(27 * 40 + 36) * 3..(27 * 40 + 36) * 3 + 3].copy_from_slice(&[9, 9, 9]);
        let mut out = Vec::new();
        let mut writer = Box::new(Writer::new(&mut out, 40, 30).unwrap());
        writer.frame(&a, 33).unwrap();
        writer.frame(&b, 33).unwrap();
        writer.frame(&b, 33).unwrap();
        writer.finish().unwrap();

        let frames = decode(out);
        assert_eq!(frames[0].2, 40 * 30, "the first frame is the whole picture");
        assert_eq!(frames[1].2, 34 * 26, "the rectangle between the two pixels");
        assert_eq!(frames[2].2, 1, "nothing changed: one transparent pixel");
        assert_eq!(frames[1].0, b.rgb);
        assert_eq!(frames[2].0, b.rgb);
    }

    #[test]
    fn a_palette_is_kept_while_it_serves_and_rebuilt_when_it_does_not() {
        // More than 256 colours: a full sweep of reds. NeuQuant has to pick.
        let many: Vec<[u8; 3]> = (0..=255)
            .map(|r| [r, 0, 0])
            .chain((0..=255).map(|g| [0, g, 0]))
            .collect();
        let a = frame_of(64, 64, &many);
        let mut palette = Palette::new();
        let mut indices = Vec::new();
        assert!(!palette.map(&a.rgb, &mut indices), "too many for exact");
        palette.rebuild(&a.rgb, &mut indices);
        let built = palette.colors.clone();
        assert_eq!(built.len(), PALETTE_SIZE);

        // The same scene again, and a few stray pixels of something new:
        // served by the palette as it stands.
        let mut b = a.clone();
        b.rgb[..6].copy_from_slice(&[0, 0, 255, 0, 0, 255]);
        assert!(palette.map(&b.rgb, &mut indices));
        assert_eq!(palette.colors, built, "nothing moved");

        // A whole patch of blue, which the red and green palette cannot
        // show: time to rebuild.
        let c = frame_of(64, 64, &[[0, 0, 255], [0, 40, 255], [255, 0, 0]]);
        assert!(!palette.map(&c.rgb, &mut indices));
        palette.rebuild(&c.rgb, &mut indices);
        assert!(
            palette
                .colors
                .iter()
                .any(|&c| distance(c, &[0, 0, 255]) <= TOLERANCE),
            "blue is in the new palette"
        );
    }
}
