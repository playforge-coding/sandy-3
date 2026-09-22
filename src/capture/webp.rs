//! Lossless WebP screenshots and animated WebP recordings, through libwebp.
//!
//! `libwebp-sys` builds Google's libwebp from source and exposes its C API,
//! and this is the little that is needed of it. A still is one call. A
//! recording goes through `WebPAnimEncoder`, which takes a frame at a time
//! and compresses it as it comes, keeping only the compressed bytes until
//! the file is put together at the end; it also works out for itself which
//! part of each frame changed and stores only that, so a world that is
//! mostly still costs little per frame.

use std::ffi::{CStr, c_int, c_void};
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::ptr;

use libwebp_sys as webp;

use super::Frame;

/// How hard libwebp tries on each recorded frame, from zero to nine. Two is
/// where the curve bends: it is a third of the time of the slowest setting
/// for a file three percent bigger, and keeps well within a frame at thirty
/// a second.
const RECORDING_EFFORT: c_int = 2;

/// Write `frame` to `w` as a lossless WebP.
pub fn write_still<W: Write>(mut w: W, frame: &Frame) -> io::Result<()> {
    let mut out: *mut u8 = ptr::null_mut();
    // SAFETY: the pointer, size and stride describe `frame.rgb` exactly, and
    // libwebp hands back a buffer it allocated, which is freed below once it
    // has been written out.
    let size = unsafe {
        webp::WebPEncodeLosslessRGB(
            frame.rgb.as_ptr(),
            frame.width as c_int,
            frame.height as c_int,
            (frame.width * 3) as c_int,
            &mut out,
        )
    };
    if size == 0 || out.is_null() {
        return Err(io::Error::other("libwebp could not encode the image"));
    }
    let result = w.write_all(unsafe { std::slice::from_raw_parts(out, size) });
    unsafe { webp::WebPFree(out as *mut c_void) };
    result
}

/// The encoder handle, so that it is deleted whichever way the writer ends.
struct Encoder(*mut webp::WebPAnimEncoder);

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: the handle came from `WebPAnimEncoderNewInternal` and is
        // deleted exactly once, here.
        unsafe { webp::WebPAnimEncoderDelete(self.0) };
    }
}

/// An animated lossless WebP, one frame at a time.
pub struct AnimationWriter<W: Write> {
    w: W,
    encoder: Encoder,
    config: webp::WebPConfig,
    width: u32,
    height: u32,
    /// When the next frame starts, in milliseconds from the first.
    timestamp: u32,
}

impl<W: Write> AnimationWriter<W> {
    pub fn new(w: W, width: u32, height: u32) -> io::Result<Self> {
        // SAFETY: each init function fills the struct it is given, and says
        // so with a non-zero return, which is checked before it is used. The
        // ABI version is passed so a libwebp built for a different layout
        // refuses rather than reading the struct wrong.
        let (encoder, config) = unsafe {
            let mut options = MaybeUninit::<webp::WebPAnimEncoderOptions>::uninit();
            if webp::WebPAnimEncoderOptionsInitInternal(
                options.as_mut_ptr(),
                webp::WEBP_MUX_ABI_VERSION as c_int,
            ) == 0
            {
                return Err(io::Error::other("libwebp's mux ABI does not match"));
            }
            let mut options = options.assume_init();
            options.anim_params.loop_count = 0; // forever

            let encoder = webp::WebPAnimEncoderNewInternal(
                width as c_int,
                height as c_int,
                &options,
                webp::WEBP_MUX_ABI_VERSION as c_int,
            );
            if encoder.is_null() {
                return Err(io::Error::other(
                    "libwebp could not make an animation encoder",
                ));
            }
            let encoder = Encoder(encoder);

            let mut config = MaybeUninit::<webp::WebPConfig>::uninit();
            if webp::WebPConfigInitInternal(
                config.as_mut_ptr(),
                webp::WebPPreset::WEBP_PRESET_DEFAULT,
                75.0,
                webp::WEBP_ENCODER_ABI_VERSION as c_int,
            ) == 0
            {
                return Err(io::Error::other("libwebp's encoder ABI does not match"));
            }
            let mut config = config.assume_init();
            if webp::WebPConfigLosslessPreset(&mut config, RECORDING_EFFORT) == 0
                || webp::WebPValidateConfig(&config) == 0
            {
                return Err(io::Error::other("libwebp rejected the lossless settings"));
            }
            (encoder, config)
        };
        Ok(AnimationWriter {
            w,
            encoder,
            config,
            width,
            height,
            timestamp: 0,
        })
    }

    /// What the encoder has to say about its last failure.
    fn error(&self) -> io::Error {
        // SAFETY: the string is libwebp's own, static, and NUL-terminated.
        let message = unsafe {
            let ptr = webp::WebPAnimEncoderGetError(self.encoder.0);
            if ptr.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        io::Error::other(format!("libwebp: {message}"))
    }
}

impl<W: Write> super::AnimationWriter for AnimationWriter<W> {
    fn frame(&mut self, frame: &Frame, duration_ms: u32) -> io::Result<()> {
        assert_eq!((frame.width, frame.height), (self.width, self.height));
        // SAFETY: the picture is initialised before use, the import reads
        // exactly `frame.rgb` at the stride given, and the picture is freed
        // after the encoder has taken its copy, on every path.
        let added = unsafe {
            let mut picture = MaybeUninit::<webp::WebPPicture>::uninit();
            if webp::WebPPictureInitInternal(
                picture.as_mut_ptr(),
                webp::WEBP_ENCODER_ABI_VERSION as c_int,
            ) == 0
            {
                return Err(io::Error::other("libwebp's encoder ABI does not match"));
            }
            let mut picture = picture.assume_init();
            picture.use_argb = 1; // lossless works on ARGB, not YUV
            picture.width = self.width as c_int;
            picture.height = self.height as c_int;
            if webp::WebPPictureImportRGB(
                &mut picture,
                frame.rgb.as_ptr(),
                (self.width * 3) as c_int,
            ) == 0
            {
                webp::WebPPictureFree(&mut picture);
                return Err(io::Error::other("libwebp could not take the frame"));
            }
            let added = webp::WebPAnimEncoderAdd(
                self.encoder.0,
                &mut picture,
                self.timestamp as c_int,
                &self.config,
            );
            webp::WebPPictureFree(&mut picture);
            added != 0
        };
        if !added {
            return Err(self.error());
        }
        self.timestamp = self.timestamp.saturating_add(duration_ms);
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> io::Result<()> {
        // SAFETY: adding a null frame is how libwebp is told when the last
        // frame ends. The assembled bytes are libwebp's, read once and freed.
        unsafe {
            if webp::WebPAnimEncoderAdd(
                self.encoder.0,
                ptr::null_mut(),
                self.timestamp as c_int,
                ptr::null(),
            ) == 0
            {
                return Err(self.error());
            }
            let mut data = webp::WebPData {
                bytes: ptr::null(),
                size: 0,
            };
            if webp::WebPAnimEncoderAssemble(self.encoder.0, &mut data) == 0 {
                return Err(self.error());
            }
            let result = self
                .w
                .write_all(std::slice::from_raw_parts(data.bytes, data.size));
            webp::WebPFree(data.bytes as *mut c_void);
            result?;
        }
        self.w.flush()
    }
}

/// Decoding, for the tests: libwebp's own decoders, which are the reference
/// ones. (The pure-Rust `image-webp` was tried and gets the blended frames
/// of an animation wrong by a pixel and a level.)
#[cfg(test)]
pub(super) mod tests {
    use super::super::AnimationWriter as _;
    use super::super::tests::test_frame;
    use super::*;

    /// A still as RGB: its width, height and pixels.
    pub fn decode_still(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
        let (mut width, mut height): (c_int, c_int) = (0, 0);
        // SAFETY: libwebp reads `bytes` and returns a buffer of the size it
        // reports, freed once copied.
        unsafe {
            let ptr = webp::WebPDecodeRGB(bytes.as_ptr(), bytes.len(), &mut width, &mut height);
            assert!(!ptr.is_null(), "libwebp could not decode the still");
            let rgb = std::slice::from_raw_parts(ptr, (width * height * 3) as usize).to_vec();
            webp::WebPFree(ptr as *mut c_void);
            (width as u32, height as u32, rgb)
        }
    }

    /// A decoded frame: its RGB pixels and the time it ends at, in
    /// milliseconds from the start.
    type DecodedFrame = (Vec<u8>, u32);

    /// An animation as a viewer would show it: the canvas size and the loop
    /// count, then each frame.
    fn decode_animation(bytes: &[u8]) -> ((u32, u32), u32, Vec<DecodedFrame>) {
        // SAFETY: the same init-and-check pattern as the encoder, and the
        // decoder owns the frame buffer it hands out until the next call, so
        // each is copied before moving on.
        unsafe {
            let mut options = MaybeUninit::<webp::WebPAnimDecoderOptions>::uninit();
            assert_ne!(
                webp::WebPAnimDecoderOptionsInitInternal(
                    options.as_mut_ptr(),
                    webp::WEBP_DEMUX_ABI_VERSION as c_int,
                ),
                0
            );
            let mut options = options.assume_init();
            options.color_mode = webp::WEBP_CSP_MODE::MODE_RGBA;
            let data = webp::WebPData {
                bytes: bytes.as_ptr(),
                size: bytes.len(),
            };
            let decoder = webp::WebPAnimDecoderNewInternal(
                &data,
                &options,
                webp::WEBP_DEMUX_ABI_VERSION as c_int,
            );
            assert!(!decoder.is_null(), "libwebp could not open the animation");
            let mut info = MaybeUninit::<webp::WebPAnimInfo>::uninit();
            assert_ne!(webp::WebPAnimDecoderGetInfo(decoder, info.as_mut_ptr()), 0);
            let info = info.assume_init();
            let mut frames = Vec::new();
            while webp::WebPAnimDecoderHasMoreFrames(decoder) != 0 {
                let mut buf: *mut u8 = ptr::null_mut();
                let mut timestamp: c_int = 0;
                assert_ne!(
                    webp::WebPAnimDecoderGetNext(decoder, &mut buf, &mut timestamp),
                    0
                );
                let rgba = std::slice::from_raw_parts(
                    buf,
                    (info.canvas_width * info.canvas_height * 4) as usize,
                );
                let rgb = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                frames.push((rgb, timestamp as u32));
            }
            webp::WebPAnimDecoderDelete(decoder);
            (
                (info.canvas_width, info.canvas_height),
                info.loop_count,
                frames,
            )
        }
    }

    #[test]
    fn an_animation_round_trips_with_its_durations() {
        let frames: Vec<Frame> = (0..3).map(|i| test_frame(29, 17, i * 60)).collect();
        let mut out = Vec::new();
        let mut writer = Box::new(AnimationWriter::new(&mut out, 29, 17).unwrap());
        for (frame, ms) in frames.iter().zip([33, 34, 500]) {
            writer.frame(frame, ms).unwrap();
        }
        writer.finish().unwrap();

        let (canvas, loops, decoded) = decode_animation(&out);
        assert_eq!(canvas, (29, 17));
        assert_eq!(loops, 0, "forever");
        assert_eq!(decoded.len(), 3);
        // Each frame ends where the next begins: 33, 67, 567 ms.
        for ((rgb, ends_at), (frame, end)) in decoded.iter().zip(frames.iter().zip([33, 67, 567])) {
            assert_eq!(*ends_at, end);
            assert_eq!(rgb, &frame.rgb);
        }
    }
}
