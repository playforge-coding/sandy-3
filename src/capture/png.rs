//! PNG screenshots and animated PNG recordings, written chunk by chunk.
//!
//! A PNG is a signature and a run of chunks, each a length, a four-letter
//! name, the data and a CRC. A still is `IHDR`, one `IDAT` holding the
//! deflated pixels, and `IEND`; one with 256 colours or fewer gets a `PLTE`
//! and one byte per pixel instead of three, which is a third of the work for
//! deflate and the reason a GIF of the same picture is small. An animated
//! one adds `acTL`, which says how many frames there are, and puts an `fcTL`
//! before each frame saying where it goes and how long it is shown; the
//! first frame's pixels stay in `IDAT`, so a viewer that knows nothing of
//! animation shows it as a still, and the rest go in `fdAT`, which is `IDAT`
//! with a sequence number in front.
//!
//! The animation cannot use a palette, because a PNG has one for the whole
//! file and a recording can pass 256 colours as materials come into the
//! scene. It saves its bytes another way: each frame after the first is
//! only the rectangle that differs from the frame before, with the pixels
//! inside it that did not change left transparent, so a mostly still world
//! deflates to almost nothing a frame.
//!
//! This is written here rather than with the `png` crate because that crate
//! wants the frame count before the first frame is written and refuses to
//! finish a file that was promised more, and a recording does not know how
//! long it will be until Stop is pressed. Writing the chunks directly means
//! `acTL` can be written with a count of zero and filled in at the end, with
//! a seek back to it, and every frame goes to disk the moment it arrives.

use std::collections::HashMap;
use std::io::{self, Seek, SeekFrom, Write};

use flate2::Compression;
use flate2::write::ZlibEncoder;

use super::{Frame, Rect};

const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// The PNG filters applied to a row before deflate. Sub is each byte minus
/// the same byte of the pixel before, so a run of one colour deflates as a
/// run of zeros; it suits colour samples, where neighbours are alike. None
/// leaves the row as it is, which suits palette indices: an index is a name
/// rather than a quantity, and the difference of two is noise.
const FILTER_NONE: u8 = 0;
const FILTER_SUB: u8 = 1;

/// PNG colour types: a palette index, RGB, and RGB with alpha.
const INDEXED: u8 = 3;
const RGB: u8 = 2;
const RGBA: u8 = 6;

/// Write one chunk: length, name, data and the CRC over name and data.
fn chunk<W: Write>(w: &mut W, name: &[u8; 4], data: &[u8]) -> io::Result<()> {
    w.write_all(&(data.len() as u32).to_be_bytes())?;
    w.write_all(name)?;
    w.write_all(data)?;
    let mut crc = crc32fast::Hasher::new();
    crc.update(name);
    crc.update(data);
    w.write_all(&crc.finalize().to_be_bytes())
}

/// The header chunk's data: eight bits a sample, the given colour type, no
/// interlacing.
fn ihdr(width: u32, height: u32, color_type: u8) -> [u8; 13] {
    let mut data = [0u8; 13];
    data[0..4].copy_from_slice(&width.to_be_bytes());
    data[4..8].copy_from_slice(&height.to_be_bytes());
    data[8] = 8; // bit depth
    data[9] = color_type;
    // Compression method, filter method and interlace method are all zero.
    data
}

/// The animation control chunk's data: the frame count, and how many times
/// to play, where zero is forever.
fn actl(frames: u32) -> [u8; 8] {
    let mut data = [0u8; 8];
    data[0..4].copy_from_slice(&frames.to_be_bytes());
    data
}

/// Filter and deflate an image of `width` pixels a row and `bpp` bytes a
/// pixel, which is what goes in `IDAT` and, after a sequence number, in
/// `fdAT`.
fn compress(
    pixels: &[u8],
    width: usize,
    bpp: usize,
    filter: u8,
    level: Compression,
) -> io::Result<Vec<u8>> {
    let row_bytes = width * bpp;
    let mut filtered = Vec::with_capacity(pixels.len() + pixels.len() / row_bytes.max(1));
    for row in pixels.chunks_exact(row_bytes) {
        filtered.push(filter);
        if filter == FILTER_NONE {
            filtered.extend_from_slice(row);
            continue;
        }
        filtered.extend_from_slice(&row[..bpp]);
        for i in bpp..row.len() {
            filtered.push(row[i].wrapping_sub(row[i - bpp]));
        }
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), level);
    encoder.write_all(&filtered)?;
    encoder.finish()
}

/// The frame as a palette and one index per pixel, if it has 256 colours or
/// fewer, in the order the colours were first met.
fn indexed(frame: &Frame) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut palette = Vec::new();
    let mut index_of = HashMap::new();
    let mut indices = Vec::with_capacity(frame.rgb.len() / 3);
    for px in frame.rgb.as_chunks::<3>().0 {
        let next = index_of.len();
        let index = *index_of.entry(*px).or_insert(next);
        if index == next {
            if next == 256 {
                return None;
            }
            palette.extend_from_slice(px);
        }
        indices.push(index as u8);
    }
    Some((palette, indices))
}

/// Write `frame` to `w` as a still PNG.
pub fn write_still<W: Write>(mut w: W, frame: &Frame) -> io::Result<()> {
    w.write_all(&SIGNATURE)?;
    // A screenshot is one image, so it can afford the slowest, smallest
    // deflate.
    let level = Compression::best();
    match indexed(frame) {
        Some((palette, indices)) => {
            chunk(&mut w, b"IHDR", &ihdr(frame.width, frame.height, INDEXED))?;
            chunk(&mut w, b"PLTE", &palette)?;
            let data = compress(&indices, frame.width as usize, 1, FILTER_NONE, level)?;
            chunk(&mut w, b"IDAT", &data)?;
        }
        None => {
            chunk(&mut w, b"IHDR", &ihdr(frame.width, frame.height, RGB))?;
            let data = compress(&frame.rgb, frame.width as usize, 3, FILTER_SUB, level)?;
            chunk(&mut w, b"IDAT", &data)?;
        }
    }
    chunk(&mut w, b"IEND", &[])
}

/// An animated PNG, one frame at a time.
pub struct AnimationWriter<W: Write + Seek> {
    w: W,
    width: u32,
    height: u32,
    /// Frames written so far, for `acTL` at the end.
    frames: u32,
    /// The next `fcTL` or `fdAT` sequence number. They share one count.
    sequence: u32,
    /// Where the `acTL` chunk starts, to come back to.
    actl_at: u64,
    /// The last frame written, to find what the next one changes.
    previous: Option<Vec<u8>>,
}

impl<W: Write + Seek> AnimationWriter<W> {
    pub fn new(mut w: W, width: u32, height: u32) -> io::Result<Self> {
        w.write_all(&SIGNATURE)?;
        chunk(&mut w, b"IHDR", &ihdr(width, height, RGBA))?;
        let actl_at = w.stream_position()?;
        chunk(&mut w, b"acTL", &actl(0))?;
        Ok(AnimationWriter {
            w,
            width,
            height,
            frames: 0,
            sequence: 0,
            actl_at,
            previous: None,
        })
    }

    /// The frame control chunk's data: the sequence number, where the frame
    /// goes and how big it is, and its duration as a fraction of a second in
    /// milliseconds.
    fn fctl(&mut self, rect: Rect, duration_ms: u32) -> [u8; 26] {
        let mut data = [0u8; 26];
        data[0..4].copy_from_slice(&self.sequence.to_be_bytes());
        data[4..8].copy_from_slice(&rect.width.to_be_bytes());
        data[8..12].copy_from_slice(&rect.height.to_be_bytes());
        data[12..16].copy_from_slice(&rect.x.to_be_bytes());
        data[16..20].copy_from_slice(&rect.y.to_be_bytes());
        let num = duration_ms.min(u16::MAX as u32) as u16;
        data[20..22].copy_from_slice(&num.to_be_bytes());
        data[22..24].copy_from_slice(&1000u16.to_be_bytes());
        // dispose_op: none, so the canvas keeps what was drawn. blend_op:
        // over, so the transparent pixels of a delta leave it alone.
        data[25] = 1;
        self.sequence += 1;
        data
    }
}

impl<W: Write + Seek> super::AnimationWriter for AnimationWriter<W> {
    fn frame(&mut self, frame: &Frame, duration_ms: u32) -> io::Result<()> {
        assert_eq!((frame.width, frame.height), (self.width, self.height));

        // The first frame is the whole picture. Every later one is the
        // rectangle that changed, with only the changed pixels in it opaque.
        // A frame that changed nothing still has to be written to carry its
        // time, and one transparent pixel does that.
        let (rect, rgba) = match &self.previous {
            None => (
                Rect::whole(self.width, self.height),
                frame
                    .rgb
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|px| [px[0], px[1], px[2], 255])
                    .collect::<Vec<u8>>(),
            ),
            Some(previous) => match Rect::changed(previous, &frame.rgb, self.width) {
                None => (Rect::new(0, 0, 1, 1), vec![0, 0, 0, 0]),
                Some(rect) => {
                    let mut rgba = Vec::with_capacity((rect.width * rect.height * 4) as usize);
                    for (before, after) in rect.rows(previous, &frame.rgb, self.width) {
                        for (b, a) in before
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .zip(after.as_chunks::<3>().0)
                        {
                            if a == b {
                                rgba.extend_from_slice(&[0, 0, 0, 0]);
                            } else {
                                rgba.extend_from_slice(&[a[0], a[1], a[2], 255]);
                            }
                        }
                    }
                    (rect, rgba)
                }
            },
        };

        let fctl = self.fctl(rect, duration_ms);
        chunk(&mut self.w, b"fcTL", &fctl)?;
        // Frames arrive thirty a second, and a delta is mostly zeros, so the
        // default deflate keeps up and packs them well.
        let pixels = compress(
            &rgba,
            rect.width as usize,
            4,
            FILTER_SUB,
            Compression::default(),
        )?;
        if self.frames == 0 {
            chunk(&mut self.w, b"IDAT", &pixels)?;
        } else {
            let mut data = Vec::with_capacity(pixels.len() + 4);
            data.extend_from_slice(&self.sequence.to_be_bytes());
            data.extend_from_slice(&pixels);
            self.sequence += 1;
            chunk(&mut self.w, b"fdAT", &data)?;
        }
        self.frames += 1;
        match &mut self.previous {
            Some(previous) => previous.copy_from_slice(&frame.rgb),
            None => self.previous = Some(frame.rgb.clone()),
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> io::Result<()> {
        chunk(&mut self.w, b"IEND", &[])?;
        self.w.seek(SeekFrom::Start(self.actl_at))?;
        chunk(&mut self.w, b"acTL", &actl(self.frames))?;
        self.w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::super::AnimationWriter as _;
    use super::super::tests::{frame_of, test_frame};
    use super::*;
    use std::io::Cursor;

    /// Decode a still, whatever its colour type, to RGB.
    fn decode_still(bytes: Vec<u8>) -> (u32, u32, ::png::ColorType, Vec<u8>) {
        let mut decoder = ::png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(::png::Transformations::EXPAND);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info.width, info.height, reader.info().color_type, buf)
    }

    #[test]
    fn a_still_of_few_colours_is_indexed_and_of_many_is_rgb() {
        let few = frame_of(40, 30, &[[0, 0, 0], [255, 200, 20], [30, 60, 200]]);
        let mut out = Vec::new();
        write_still(&mut out, &few).unwrap();
        let (w, h, color_type, rgb) = decode_still(out);
        assert_eq!((w, h), (40, 30));
        assert_eq!(color_type, ::png::ColorType::Indexed);
        assert_eq!(rgb, few.rgb);

        // The gradient has more than 256 colours.
        let many = test_frame(40, 30, 0);
        let mut out = Vec::new();
        write_still(&mut out, &many).unwrap();
        let (_, _, color_type, rgb) = decode_still(out);
        assert_eq!(color_type, ::png::ColorType::Rgb);
        assert_eq!(rgb, many.rgb);

        // Indexed is the point. A noisy picture of two hundred colours, so
        // deflate has no pattern to lean on: one byte a pixel against three.
        let colors: Vec<[u8; 3]> = (0..200u32)
            .map(|i| {
                [
                    (i * 37 % 256) as u8,
                    (i * 91 % 256) as u8,
                    (i * 13 % 256) as u8,
                ]
            })
            .collect();
        let mut state = 0x9E37_79B9u32; // xorshift, so the picture is noise
        let rgb: Vec<u8> = (0..200 * 200)
            .flat_map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                colors[state as usize % colors.len()]
            })
            .collect();
        let noisy = Frame {
            width: 200,
            height: 200,
            rgb,
        };
        let mut out = Vec::new();
        write_still(&mut out, &noisy).unwrap();
        let indexed_size = out.len();
        let (_, _, color_type, decoded) = decode_still(out);
        assert_eq!(color_type, ::png::ColorType::Indexed);
        assert_eq!(decoded, noisy.rgb);
        let rgb_size = compress(&noisy.rgb, 200, 3, FILTER_SUB, Compression::best())
            .unwrap()
            .len();
        assert!(
            indexed_size < rgb_size,
            "indexed {indexed_size} against rgb {rgb_size}"
        );
    }

    #[test]
    fn an_animation_round_trips_through_its_deltas() {
        // A gradient, then a change in one corner, then the same again, then
        // a different corner.
        let a = test_frame(31, 19, 0);
        let mut b = a.clone();
        b.rgb[..9].copy_from_slice(&[9; 9]);
        let c = b.clone();
        let mut d = c.clone();
        let last = d.rgb.len();
        d.rgb[last - 3..].copy_from_slice(&[7, 8, 9]);
        let frames = [&a, &b, &c, &d];

        let mut out = Cursor::new(Vec::new());
        let mut writer = Box::new(AnimationWriter::new(&mut out, 31, 19).unwrap());
        for (frame, ms) in frames.iter().zip([33, 34, 500, 20]) {
            writer.frame(frame, ms).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = ::png::Decoder::new(Cursor::new(out.into_inner()))
            .read_info()
            .unwrap();
        let control = reader.info().animation_control().unwrap();
        assert_eq!(control.num_frames, 4, "the count patched in at the end");
        assert_eq!(control.num_plays, 0, "forever");
        assert_eq!(reader.info().color_type, ::png::ColorType::Rgba);

        // Composite each frame over the last the way a viewer would: an
        // opaque pixel replaces, a transparent one leaves.
        let mut canvas = vec![0u8; 31 * 19 * 3];
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        for (i, (frame, ms)) in frames.iter().zip([33u16, 34, 500, 20]).enumerate() {
            let fctl = *if i == 0 {
                reader.info().frame_control().unwrap()
            } else {
                reader.next_frame_info().unwrap()
            };
            assert_eq!((fctl.delay_num, fctl.delay_den), (ms, 1000));
            if i == 0 {
                assert_eq!((fctl.width, fctl.height), (31, 19));
            } else {
                assert!(
                    fctl.width * fctl.height <= 9,
                    "a delta, not the whole frame"
                );
            }
            let info = reader.next_frame(&mut buf).unwrap();
            assert_eq!(info.buffer_size(), (fctl.width * fctl.height * 4) as usize);
            for (row, line) in buf[..info.buffer_size()]
                .chunks_exact(fctl.width as usize * 4)
                .enumerate()
            {
                for (col, px) in line.as_chunks::<4>().0.iter().enumerate() {
                    if px[3] == 255 {
                        let at = ((fctl.y_offset + row as u32) * 31 + fctl.x_offset + col as u32)
                            as usize
                            * 3;
                        canvas[at..at + 3].copy_from_slice(&px[..3]);
                    }
                }
            }
            assert_eq!(canvas, frame.rgb, "frame {i}");
        }
        assert!(reader.next_frame_info().is_err(), "no fifth frame");
    }
}
