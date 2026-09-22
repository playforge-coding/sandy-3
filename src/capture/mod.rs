//! Screenshots and recordings.
//!
//! Both are taken from the same place: [`crate::gpu::State::render`] draws the
//! finished scene, glow and all, a second time into an offscreen image at the
//! grid's own resolution and copies it out to a buffer the CPU can read. The
//! panel is not in it, and nor is the window's stretch, so a capture is one
//! pixel per cell whatever the window looks like. The frames come back a frame
//! or two later, since the copy is left to complete on its own rather than
//! waited for; [`Capture`] keeps a note of what each one is for and hands it
//! on when it arrives.
//!
//! Encoding is done off the main thread. A screenshot is one short job. A
//! recording is a thread that lives as long as the recording does: frames are
//! sent to it as they come back from the GPU and it writes each to the file
//! there and then, so a long recording costs no more memory than a short one.
//! The writers live in [`png`], [`gif`] and [`webp`], one per format, behind
//! [`AnimationWriter`].
//!
//! Every file goes in [`CAPTURE_DIR`], named after the local time it was taken.

mod gif;
mod png;
mod webp;

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Where captures are written, relative to the working directory. Made if it
/// is missing.
pub const CAPTURE_DIR: &str = "captures";

/// How many frames a second a recording takes. Thirty is plenty for sand, and
/// the frames a display draws between them would double the file for nothing.
/// The cadence is kept by [`Cadence`] and the real gap between the frames it
/// picks, not this nominal one, is what goes in the file.
const RECORD_FPS: f64 = 30.0;

/// A frame read back from the GPU: the composited scene at the grid's own
/// resolution, as tightly packed RGB bytes, row by row from the top.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Frame {
    /// Build a frame from the bytes wgpu copies a texture into: RGBA, with
    /// each row padded out to `row_bytes`. The padding and the alpha, which
    /// is always full, are dropped here.
    pub fn from_padded_rgba(data: &[u8], width: u32, height: u32, row_bytes: usize) -> Self {
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for row in data.chunks_exact(row_bytes).take(height as usize) {
            for px in row[..width as usize * 4].as_chunks::<4>().0 {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        Frame { width, height, rgb }
    }
}

/// A rectangle of a frame, in pixels from the top left. The GIF and PNG
/// writers use one to write only the part of a frame that changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    pub fn whole(width: u32, height: u32) -> Self {
        Rect::new(0, 0, width, height)
    }

    /// The smallest rectangle holding every pixel that differs between two
    /// RGB frames `width` pixels across, or nothing if they are the same.
    /// Rows are compared whole first, which is a fast memory compare, and
    /// only the rows that differ are looked through for where.
    pub fn changed(before: &[u8], after: &[u8], width: u32) -> Option<Rect> {
        let row_bytes = width as usize * 3;
        let (mut top, mut bottom) = (None, 0);
        let (mut left, mut right) = (width, 0);
        let rows = before
            .chunks_exact(row_bytes)
            .zip(after.chunks_exact(row_bytes));
        for (y, (b, a)) in rows.enumerate() {
            if a == b {
                continue;
            }
            top.get_or_insert(y);
            bottom = y;
            let pairs = || b.as_chunks::<3>().0.iter().zip(a.as_chunks::<3>().0);
            let first = pairs().position(|(b, a)| a != b).expect("the rows differ") as u32;
            let last = pairs().rposition(|(b, a)| a != b).expect("the rows differ") as u32;
            left = left.min(first);
            right = right.max(last);
        }
        top.map(|top| {
            Rect::new(
                left,
                top as u32,
                right - left + 1,
                (bottom - top + 1) as u32,
            )
        })
    }

    /// This rectangle's rows of two RGB frames `width` pixels across, as a
    /// pair of byte slices a row, before and after.
    pub fn rows<'a>(
        &self,
        before: &'a [u8],
        after: &'a [u8],
        width: u32,
    ) -> impl Iterator<Item = (&'a [u8], &'a [u8])> {
        let row_bytes = width as usize * 3;
        let rect = *self;
        (rect.y..rect.y + rect.height).map(move |y| {
            let start = y as usize * row_bytes + rect.x as usize * 3;
            let end = start + rect.width as usize * 3;
            (&before[start..end], &after[start..end])
        })
    }
}

/// What a screenshot is saved as.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImageFormat {
    Png,
    #[default]
    WebP,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 2] = [ImageFormat::WebP, ImageFormat::Png];

    pub fn name(self) -> &'static str {
        match self {
            ImageFormat::Png => "PNG",
            ImageFormat::WebP => "WebP",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::WebP => "webp",
        }
    }
}

/// What a recording is saved as. All three are lossless in themselves; the
/// GIF is only as true as its 256 colours a frame allow (see [`gif`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AnimationFormat {
    Gif,
    /// An animated PNG, which keeps the `.png` extension: a viewer that does
    /// not know about the animation shows the first frame.
    Apng,
    #[default]
    WebP,
}

impl AnimationFormat {
    pub const ALL: [AnimationFormat; 3] = [
        AnimationFormat::WebP,
        AnimationFormat::Gif,
        AnimationFormat::Apng,
    ];

    pub fn name(self) -> &'static str {
        match self {
            AnimationFormat::Gif => "GIF",
            AnimationFormat::Apng => "APNG",
            AnimationFormat::WebP => "WebP",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            AnimationFormat::Gif => "gif",
            AnimationFormat::Apng => "png",
            AnimationFormat::WebP => "webp",
        }
    }
}

/// One frame after another into a file, each with how long it is shown for.
/// The three formats each have one of these.
trait AnimationWriter {
    fn frame(&mut self, frame: &Frame, duration_ms: u32) -> io::Result<()>;
    /// Write whatever ends the file, and anything that could only be known
    /// once every frame was in.
    fn finish(self: Box<Self>) -> io::Result<()>;
}

/// Write one frame to `w` as a still image in `format`.
fn write_still<W: Write>(w: W, frame: &Frame, format: ImageFormat) -> io::Result<()> {
    match format {
        ImageFormat::Png => png::write_still(w, frame),
        ImageFormat::WebP => webp::write_still(w, frame),
    }
}

/// Open an animation writer for `format` on `w`, for frames of `width` by
/// `height`.
fn open_animation<W: Write + io::Seek + 'static>(
    w: W,
    format: AnimationFormat,
    width: u32,
    height: u32,
) -> io::Result<Box<dyn AnimationWriter>> {
    Ok(match format {
        AnimationFormat::Gif => Box::new(gif::Writer::new(w, width, height)?),
        AnimationFormat::Apng => Box::new(png::AnimationWriter::new(w, width, height)?),
        AnimationFormat::WebP => Box::new(webp::AnimationWriter::new(w, width, height)?),
    })
}

/// A path in `dir` that nothing is at yet, named after the local time with
/// `extension` on the end. Two captures in the same second get a number
/// after the time.
fn fresh_path(dir: &Path, extension: &str) -> PathBuf {
    let stamp = jiff::Zoned::now().strftime("%Y-%m-%d-%H%M%S").to_string();
    let first = dir.join(format!("sandy-{stamp}.{extension}"));
    if !first.exists() {
        return first;
    }
    (2..)
        .map(|n| dir.join(format!("sandy-{stamp}-{n}.{extension}")))
        .find(|path| !path.exists())
        .expect("an unbounded range has no end")
}

/// Picks which displayed frames go into a recording, so that they come
/// [`RECORD_FPS`] a second however fast the display runs.
///
/// Frames are due on a fixed grid of instants. The frame taken for each is
/// the one drawn nearest it, which is the first frame within half a frame
/// period of it: on a sixty hertz display that is every other frame, evenly,
/// where taking the first frame past the due time would give an uneven two
/// and three. A stall moves the grid on rather than banking frames to take
/// in a burst afterwards.
struct Cadence {
    interval: Duration,
    next_due: Instant,
    /// The last frame's time and how long the one before it took, which is
    /// the guess at how far off the next frame is.
    last: Instant,
    period: Duration,
}

impl Cadence {
    fn new(now: Instant) -> Self {
        Cadence {
            interval: Duration::from_secs_f64(1.0 / RECORD_FPS),
            next_due: now,
            last: now,
            period: Duration::ZERO,
        }
    }

    /// Note that a frame is being drawn at `now`, and say whether it should
    /// be taken. Taking it is [`Cadence::taken`].
    fn due(&mut self, now: Instant) -> bool {
        self.period = now.saturating_duration_since(self.last);
        self.last = now;
        now + self.period / 2 >= self.next_due
    }

    /// The frame drawn at `now` was taken: move on to the next due time. The
    /// grid holds, so a frame a few milliseconds late does not shift it,
    /// unless the frame came a whole interval or more late, which is a
    /// stall, when the next is measured from the frame instead rather than
    /// taking the frames after it in a burst.
    fn taken(&mut self, now: Instant) {
        self.next_due += self.interval;
        if now > self.next_due {
            self.next_due = now + self.interval;
        }
    }
}

/// What a frame still on its way back from the GPU is wanted for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wanted {
    /// Save it as a screenshot in this format.
    pub screenshot: Option<ImageFormat>,
    /// Add it to the recording, as the frame drawn at this instant.
    pub record: Option<Instant>,
}

/// What the encoder threads say when they are done: a line for the panel.
pub struct Report(pub String);

enum Message {
    Frame(Frame, Instant),
    Stop,
}

/// A recording in progress: the thread writing it, reached through `frames`.
struct Recording {
    frames: Sender<Message>,
    path: PathBuf,
    started: Instant,
    cadence: Cadence,
    /// Stop was pressed. The thread is told once the frames already asked
    /// of the GPU have come back, so the recording ends where it was stopped
    /// rather than a frame or two before.
    stopping: bool,
}

/// The app's side of capturing: takes the requests, decides which frames the
/// GPU should copy out, and passes the frames that come back to the threads
/// that write them.
pub struct Capture {
    dir: PathBuf,
    /// A screenshot has been asked for, in this format, and the next frame
    /// is it.
    screenshot: Option<ImageFormat>,
    recording: Option<Recording>,
    /// What each frame the GPU is still copying out is for, oldest first.
    /// They come back in the order they were asked for.
    wanted: VecDeque<Wanted>,
    reports: Receiver<Report>,
    report: Sender<Report>,
}

impl Default for Capture {
    fn default() -> Self {
        let (report, reports) = mpsc::channel();
        Capture {
            dir: PathBuf::from(CAPTURE_DIR),
            screenshot: None,
            recording: None,
            wanted: VecDeque::new(),
            reports,
            report,
        }
    }
}

impl Capture {
    /// Save the next frame as a screenshot.
    pub fn screenshot(&mut self, format: ImageFormat) {
        self.screenshot = Some(format);
    }

    /// Whether a recording is on, and for how long it has been.
    pub fn recording_for(&self, now: Instant) -> Option<Duration> {
        self.recording
            .as_ref()
            .filter(|rec| !rec.stopping)
            .map(|rec| now.saturating_duration_since(rec.started))
    }

    /// Start a recording in `format`, or stop the one that is running. Says
    /// what it did, for the panel.
    pub fn toggle_recording(&mut self, format: AnimationFormat, now: Instant) -> String {
        if let Some(rec) = &mut self.recording {
            rec.stopping = true;
            self.finish_stopping();
            return "Finishing the recording.".to_string();
        }
        if let Err(err) = std::fs::create_dir_all(&self.dir) {
            return format!("Could not make {}: {err}", self.dir.display());
        }
        let path = fresh_path(&self.dir, format.extension());
        let (tx, rx) = mpsc::channel();
        let report = self.report.clone();
        let thread_path = path.clone();
        thread::spawn(move || record(rx, thread_path, format, report));
        self.recording = Some(Recording {
            frames: tx,
            path: path.clone(),
            started: now,
            cadence: Cadence::new(now),
            stopping: false,
        });
        format!(
            "Recording to {}. Press V or Stop to finish.",
            path.display()
        )
    }

    /// Whether the frame about to be drawn at `now` should be copied out, and
    /// what for. Nothing is committed to until [`Capture::taken`] says the
    /// copy was actually made, so a frame the renderer skips is asked for
    /// again next time.
    pub fn plan(&mut self, now: Instant) -> Option<Wanted> {
        let record = match &mut self.recording {
            Some(rec) if !rec.stopping => rec.cadence.due(now).then_some(now),
            _ => None,
        };
        let wanted = Wanted {
            screenshot: self.screenshot,
            record,
        };
        (wanted.screenshot.is_some() || wanted.record.is_some()).then_some(wanted)
    }

    /// The frame [`Capture::plan`] asked for was copied out: remember what it
    /// is for until it comes back.
    pub fn taken(&mut self, wanted: Wanted) {
        if wanted.screenshot.is_some() {
            self.screenshot = None;
        }
        if let (Some(at), Some(rec)) = (wanted.record, &mut self.recording) {
            rec.cadence.taken(at);
        }
        self.wanted.push_back(wanted);
    }

    /// A frame came back from the GPU: send it wherever it was wanted.
    pub fn deliver(&mut self, frame: Frame) {
        let Some(wanted) = self.wanted.pop_front() else {
            log::warn!("a frame came back that nothing asked for");
            return;
        };
        // A send fails only if the thread has gone because the recording
        // failed, and it has already said so.
        let record = wanted
            .record
            .and_then(|at| self.recording.as_ref().map(|rec| (at, &rec.frames)));
        match (wanted.screenshot, record) {
            (Some(format), Some((at, frames))) => {
                // Wanted for both, which is the one case that costs a copy.
                let _ = frames.send(Message::Frame(frame.clone(), at));
                self.save_screenshot(frame, format);
            }
            (Some(format), None) => self.save_screenshot(frame, format),
            (None, Some((at, frames))) => {
                let _ = frames.send(Message::Frame(frame, at));
            }
            (None, None) => {}
        }
        self.finish_stopping();
    }

    /// Whatever the encoder threads have finished saying since last time.
    pub fn reports(&mut self) -> Vec<String> {
        self.reports.try_iter().map(|Report(line)| line).collect()
    }

    /// Encode `frame` on a thread and report the file it went to.
    fn save_screenshot(&self, frame: Frame, format: ImageFormat) {
        let dir = self.dir.clone();
        let report = self.report.clone();
        thread::spawn(move || {
            let result = std::fs::create_dir_all(&dir).and_then(|()| {
                let path = fresh_path(&dir, format.extension());
                let mut file = BufWriter::new(File::create(&path)?);
                write_still(&mut file, &frame, format)?;
                file.flush()?;
                Ok(path)
            });
            let line = match result {
                Ok(path) => format!("Saved {}.", path.display()),
                Err(err) => format!("Screenshot failed: {err}"),
            };
            let _ = report.send(Report(line));
        });
    }

    /// If the recording has been stopped and every frame asked for it is
    /// back, tell the thread to finish the file.
    fn finish_stopping(&mut self) {
        let Some(rec) = &self.recording else {
            return;
        };
        if !rec.stopping || self.wanted.iter().any(|w| w.record.is_some()) {
            return;
        }
        let _ = rec.frames.send(Message::Stop);
        log::info!("recording to {} stopped", rec.path.display());
        self.recording = None;
    }
}

/// The recording thread: write each frame to `path` as it arrives, until told
/// to stop, then report how it went.
///
/// A frame is shown for as long as it was until the next one, which is only
/// known once that one arrives, so each frame is written when the one after
/// it comes in. The last is given the nominal frame time.
fn record(
    frames: Receiver<Message>,
    path: PathBuf,
    format: AnimationFormat,
    report: Sender<Report>,
) {
    // When Stop arrived, to say how far the writing ran behind the frames.
    let mut stopped = Instant::now();
    let result = (|| -> io::Result<(usize, Duration)> {
        let mut writer: Option<Box<dyn AnimationWriter>> = None;
        let mut held: Option<(Frame, Instant)> = None;
        let mut count = 0;
        let mut shown = Duration::ZERO;
        loop {
            // A closed channel means the app has gone; finish the file anyway.
            let message = frames.recv().unwrap_or(Message::Stop);
            let Message::Frame(frame, at) = message else {
                stopped = Instant::now();
                break;
            };
            if let Some((prev, prev_at)) = held.replace((frame, at)) {
                let gap = at.saturating_duration_since(prev_at);
                let ms = gap.as_millis().clamp(1, u32::MAX as u128) as u32;
                let writer = match &mut writer {
                    Some(writer) => writer,
                    None => {
                        let file = BufWriter::new(File::create(&path)?);
                        writer.insert(open_animation(file, format, prev.width, prev.height)?)
                    }
                };
                writer.frame(&prev, ms)?;
                count += 1;
                shown += Duration::from_millis(ms as u64);
            }
        }
        if let Some((last, _)) = held {
            let nominal = Duration::from_secs_f64(1.0 / RECORD_FPS);
            let writer = match &mut writer {
                Some(writer) => writer,
                None => {
                    let file = BufWriter::new(File::create(&path)?);
                    writer.insert(open_animation(file, format, last.width, last.height)?)
                }
            };
            writer.frame(&last, nominal.as_millis() as u32)?;
            count += 1;
            shown += nominal;
        }
        if let Some(writer) = writer {
            writer.finish()?;
        }
        Ok((count, shown))
    })();
    let line = match result {
        Ok((0, _)) => "The recording had no frames, so nothing was saved.".to_string(),
        Ok((count, shown)) => {
            log::info!(
                "wrote {} ({count} frames), finishing {:.2}s after the recording stopped",
                path.display(),
                stopped.elapsed().as_secs_f64()
            );
            format!(
                "Saved {}: {count} frames, {:.1} s.",
                path.display(),
                shown.as_secs_f64()
            )
        }
        Err(err) => {
            // Half a file is worse than none.
            let _ = std::fs::remove_file(&path);
            format!("Recording failed: {err}")
        }
    };
    let _ = report.send(Report(line));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A small frame with a gradient, a flat patch and a stripe, so an
    /// encoder that gets rows or channels crossed shows it.
    pub fn test_frame(width: u32, height: u32, shift: u8) -> Frame {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = if y < height / 3 {
                    (x as u8, y as u8, shift)
                } else if x % 7 == 0 {
                    (200, 30, 30)
                } else {
                    (10, 20, 30u8.wrapping_add(shift))
                };
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
        Frame { width, height, rgb }
    }

    /// A frame of `width` by `height` in which pixel `i` is colour
    /// `colors[i % colors.len()]`.
    pub fn frame_of(width: u32, height: u32, colors: &[[u8; 3]]) -> Frame {
        let rgb = (0..width * height)
            .flat_map(|i| colors[i as usize % colors.len()])
            .collect();
        Frame { width, height, rgb }
    }

    #[test]
    fn the_changed_rectangle_is_the_smallest_that_holds_every_difference() {
        let before = test_frame(20, 10, 0);
        assert_eq!(Rect::changed(&before.rgb, &before.rgb, 20), None);

        let mut after = before.clone();
        let at = |x: usize, y: usize| (y * 20 + x) * 3;
        after.rgb[at(3, 2)] ^= 1;
        assert_eq!(
            Rect::changed(&before.rgb, &after.rgb, 20),
            Some(Rect::new(3, 2, 1, 1))
        );
        after.rgb[at(17, 8) + 2] ^= 1;
        let rect = Rect::changed(&before.rgb, &after.rgb, 20).unwrap();
        assert_eq!(rect, Rect::new(3, 2, 15, 7));

        // The rows come out as the rectangle's own slices.
        let rows: Vec<_> = rect.rows(&before.rgb, &after.rgb, 20).collect();
        assert_eq!(rows.len(), 7);
        assert_eq!(rows[0].0.len(), 15 * 3);
        assert_eq!(rows[0].1[0], after.rgb[at(3, 2)]);
        assert_eq!(rows[6].1[14 * 3 + 2], after.rgb[at(17, 8) + 2]);
    }

    #[test]
    fn padding_and_alpha_are_stripped_from_a_readback() {
        // Two rows of three pixels, each row padded to sixteen bytes.
        let mut data = vec![0u8; 32];
        data[0..12].copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255]);
        data[16..28].copy_from_slice(&[10, 11, 12, 255, 13, 14, 15, 255, 16, 17, 18, 255]);
        let frame = Frame::from_padded_rgba(&data, 3, 2, 16);
        assert_eq!(frame.width, 3);
        assert_eq!(frame.height, 2);
        assert_eq!(
            frame.rgb,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }

    #[test]
    fn a_still_in_either_format_comes_back_as_it_went_in() {
        let frame = test_frame(37, 23, 5);
        for format in ImageFormat::ALL {
            let mut out = Cursor::new(Vec::new());
            write_still(&mut out, &frame, format).unwrap();
            let bytes = out.into_inner();
            let decoded = match format {
                ImageFormat::Png => {
                    let mut reader = ::png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
                    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
                    let info = reader.next_frame(&mut buf).unwrap();
                    assert_eq!((info.width, info.height), (37, 23));
                    assert_eq!(info.color_type, ::png::ColorType::Rgb);
                    buf.truncate(info.buffer_size());
                    buf
                }
                ImageFormat::WebP => {
                    let (width, height, rgb) = webp::tests::decode_still(&bytes);
                    assert_eq!((width, height), (37, 23));
                    rgb
                }
            };
            assert_eq!(decoded, frame.rgb, "{}", format.name());
        }
    }

    #[test]
    fn a_fresh_path_never_lands_on_an_existing_file() {
        let dir = std::env::temp_dir().join(format!("sandy-capture-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let first = fresh_path(&dir, "webp");
        assert_eq!(first.extension().unwrap(), "webp");
        assert!(
            first
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("sandy-")
        );
        std::fs::write(&first, b"").unwrap();
        let second = fresh_path(&dir, "webp");
        assert_ne!(first, second);
        std::fs::write(&second, b"").unwrap();
        let third = fresh_path(&dir, "webp");
        assert_ne!(third, first);
        assert_ne!(third, second);
        // A different extension is a different file, so the plain name is
        // free again.
        assert_eq!(fresh_path(&dir, "png").file_stem(), first.file_stem());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cadence_takes_every_other_frame_of_sixty_a_second() {
        let start = Instant::now();
        let frame = Duration::from_secs_f64(1.0 / 60.0);
        let mut cadence = Cadence::new(start);
        let mut taken = Vec::new();
        for i in 0..60 {
            let now = start + frame * i;
            if cadence.due(now) {
                cadence.taken(now);
                taken.push(i);
            }
        }
        // Thirty of sixty, and evenly: every second frame.
        assert_eq!(taken.len(), 30);
        assert!(taken.windows(2).all(|w| w[1] - w[0] == 2), "{taken:?}");
    }

    #[test]
    fn the_cadence_takes_the_nearest_frame_at_odd_rates_and_skips_a_stall() {
        let start = Instant::now();
        let frame = Duration::from_secs_f64(1.0 / 144.0);
        let mut cadence = Cadence::new(start);
        let mut taken = Vec::new();
        for i in 0..144 {
            let now = start + frame * i;
            if cadence.due(now) {
                cadence.taken(now);
                taken.push(i);
            }
        }
        // Thirty a second, each within half a frame of its due time.
        assert_eq!(taken.len(), 30);
        assert!(
            taken.windows(2).all(|w| (4..=5).contains(&(w[1] - w[0]))),
            "{taken:?}"
        );

        // A half-second stall: one frame is taken when it ends, not fifteen.
        let mut cadence = Cadence::new(start);
        assert!(cadence.due(start));
        cadence.taken(start);
        let after = start + Duration::from_millis(500);
        assert!(cadence.due(after));
        cadence.taken(after);
        let next = after + Duration::from_secs_f64(1.0 / 60.0);
        assert!(!cadence.due(next), "no burst of catch-up frames");
    }

    #[test]
    fn a_planned_frame_is_only_owed_once_taken() {
        let now = Instant::now();
        let mut capture = Capture::default();
        assert!(capture.plan(now).is_none());
        capture.screenshot(ImageFormat::Png);
        let wanted = capture.plan(now).unwrap();
        assert_eq!(wanted.screenshot, Some(ImageFormat::Png));
        assert!(wanted.record.is_none());
        // Not taken: it is still wanted next frame.
        assert!(capture.plan(now).is_some());
        capture.taken(wanted);
        assert!(capture.plan(now).is_none());
        assert_eq!(capture.wanted.len(), 1);
    }
}
