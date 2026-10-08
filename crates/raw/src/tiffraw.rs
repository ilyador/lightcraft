//! Reading the pixel data of a TIFF IFD (strips or tiles; uncompressed, lossless JPEG, Deflate), in parallel.

use crate::unpack::*;
use crate::{MAX_SAMPLES, RawData, RawError, Result, ljpeg};
use lightcraft_tiff::image::{Chunk, ImageInfo, chunk_bytes};
use lightcraft_tiff::{ByteOrder, tags::compression as comp};
use rayon::prelude::*;

/// Bit packing for uncompressed integer data with bit depths other than 8/16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // vendor decoders use the other variants
pub enum Packing {
    /// TIFF standard: MSB-first, rows start on byte boundaries.
    Msb,
    /// LSB-first little-endian bit stream (some vendor raws), rows byte aligned.
    Lsb,
    /// Each sample stored in 16 bits in file byte order, regardless of `bits`.
    Word16,
}

enum ChunkPx {
    U16(Vec<u16>),
    F32(Vec<f32>),
}

/// The checks [`read_image`] makes before decoding anything (sample format, size limits, data
/// present); returns the number of samples.
pub fn check_image(data: &[u8], info: &ImageInfo) -> Result<usize> {
    let (w, h) = (info.width as usize, info.height as usize);
    let cpp = info.samples_per_pixel as usize;
    let total = w.checked_mul(h).and_then(|v| v.checked_mul(cpp)).ok_or(RawError::Limit("image too large"))?;
    if total > MAX_SAMPLES || total == 0 {
        return Err(RawError::Limit("image too large"));
    }
    let bits = info.bits() as u32;
    let float = info.sample_format == 3;
    if float && !matches!(bits, 16 | 24 | 32) {
        return Err(RawError::Unsupported(format!("{bits}-bit floating point samples")));
    }
    if !float && !(1..=16).contains(&bits) {
        return Err(RawError::Unsupported(format!("{bits}-bit integer samples")));
    }
    let chunks = info.chunks(data.len() as u64);
    if chunks.is_empty() {
        return Err(RawError::Corrupt("no image data chunks".into()));
    }
    if info.compression == comp::JPEG_XL || info.new_subfile_type == 16 {
        let (across, down) = info.grid();
        let planes = if info.planar == 2 { info.samples_per_pixel as usize } else { 1 };
        let expected = (across as usize).checked_mul(down as usize).and_then(|n| n.checked_mul(planes));
        if expected != Some(chunks.len()) {
            return Err(RawError::Corrupt("Enhanced image has missing data chunks".into()));
        }
    }
    // Plausibility bound before allocating: no supported coding stores more than ~2000 samples per byte
    // (Deflate of constant data is the extreme), so tiny files cannot trigger huge allocations.
    let available: u64 = chunks.iter().map(|c| chunk_bytes(data, c).map_or(0, |s| s.len() as u64)).sum();
    if available.saturating_mul(2048) < total as u64 {
        return Err(RawError::Corrupt(format!("{available} bytes of image data cannot hold {total} samples")));
    }
    Ok(total)
}

/// Read the samples of `info` in `mode`: [`Mode::Header`] only checks them ([`check_image`]) and
/// returns no samples (floating-point data as an empty `F32`).
pub(crate) fn read_image_in(mode: crate::Mode, data: &[u8], info: &ImageInfo, order: ByteOrder, packing: Packing) -> Result<RawData> {
    match mode {
        crate::Mode::Full => read_image(data, info, order, packing),
        crate::Mode::Header => {
            check_image(data, info)?;
            // a lossless-JPEG layout the decoder rejects (subsampled components, e.g. Sony's
            // lossless M/S sizes) must fail here too, as the full decode will
            if info.compression == 7
                && let Some(src) = info.chunks(data.len() as u64).first().and_then(|c| chunk_bytes(data, c))
            {
                ljpeg::frame_info(src)?;
            }
            Ok(if info.sample_format == 3 { RawData::F32(Vec::new()) } else { RawData::U16(Vec::new()) })
        }
    }
}

/// Decode all chunks of `info` into one buffer of `width × height × cpp` samples.
pub fn read_image(data: &[u8], info: &ImageInfo, order: ByteOrder, packing: Packing) -> Result<RawData> {
    let (w, h) = (info.width as usize, info.height as usize);
    let total = check_image(data, info)?;
    let cpp = info.samples_per_pixel as usize;
    let bits = info.bits() as u32;
    let float = info.sample_format == 3;
    let chunks = info.chunks(data.len() as u64);
    let planar = info.planar == 2 && cpp > 1;
    let ccpp = if planar { 1 } else { cpp };
    let decoded: Vec<Result<(Chunk, ChunkPx)>> =
        chunks.par_iter().map(|c| decode_chunk(data, info, c, order, packing, ccpp, bits, float).map(|p| (*c, p))).collect();
    let mut out_u16 = if float { Vec::new() } else { vec![0u16; total] };
    let mut out_f32 = if float { vec![0f32; total] } else { Vec::new() };
    let mut ok = 0usize;
    let mut first_err = None;
    for r in decoded {
        let (c, px) = match r {
            Ok(v) => v,
            Err(e) => {
                if info.compression == comp::JPEG_XL || info.new_subfile_type == 16 {
                    return Err(e);
                }
                first_err.get_or_insert(e);
                continue;
            }
        };
        ok += 1;
        let (cw, ch) = (c.width as usize, c.height as usize);
        let (x0, y0) = (c.x as usize, c.y as usize);
        if x0 >= w || y0 >= h {
            continue;
        }
        let copy_w = cw.min(w - x0);
        let copy_h = ch.min(h - y0);
        for y in 0..copy_h {
            for x in 0..copy_w {
                for s in 0..ccpp {
                    let src = (y * cw + x) * ccpp + s;
                    let dst_s = if planar { c.plane as usize } else { s };
                    if dst_s >= cpp {
                        continue;
                    }
                    let dst = ((y0 + y) * w + x0 + x) * cpp + dst_s;
                    match &px {
                        ChunkPx::U16(v) => {
                            if let (Some(o), Some(&s)) = (out_u16.get_mut(dst), v.get(src)) {
                                *o = s;
                            }
                        }
                        ChunkPx::F32(v) => {
                            if let (Some(o), Some(&s)) = (out_f32.get_mut(dst), v.get(src)) {
                                *o = s;
                            }
                        }
                    }
                }
            }
        }
    }
    if ok == 0 {
        return Err(first_err.unwrap_or_else(|| RawError::Corrupt("no decodable chunks".into())));
    }
    Ok(if float { RawData::F32(out_f32) } else { RawData::U16(out_u16) })
}

#[allow(clippy::too_many_arguments)]
fn decode_chunk(data: &[u8], info: &ImageInfo, c: &Chunk, order: ByteOrder, packing: Packing, cpp: usize, bits: u32, float: bool) -> Result<ChunkPx> {
    let src = chunk_bytes(data, c).ok_or_else(|| RawError::Corrupt("chunk offset past end of file".into()))?;
    let (cw, ch) = (c.width as usize, c.height as usize);
    let n = cw.checked_mul(ch).and_then(|v| v.checked_mul(cpp)).ok_or(RawError::Limit("chunk too large"))?;
    if n > MAX_SAMPLES {
        return Err(RawError::Limit("chunk too large"));
    }
    match info.compression {
        comp::NONE => unpack_chunk(src, order, packing, cw, ch, cpp, bits, float, 1),
        comp::JPEG => {
            let f = ljpeg::decode(src, n.saturating_mul(2).max(1 << 16))?;
            if f.data.len() < n {
                // Some writers encode edge tiles smaller than nominal: accept a frame that covers the rows it has.
                if f.data.is_empty() || f.data.len() % (cw * cpp) != 0 {
                    return Err(RawError::Corrupt(format!("lossless JPEG tile has {} samples, expected {n}", f.data.len())));
                }
            }
            let mut v = f.data;
            v.resize(n, 0);
            Ok(ChunkPx::U16(v))
        }
        comp::LOSSY_JPEG => {
            // lossy DNG: each chunk is a baseline (DCT) JPEG of 8-bit samples
            let (px, jw, jh, jc) = lossy_jpeg(src, cpp)?;
            let mut v = vec![0u16; n];
            for y in 0..ch.min(jh) {
                for x in 0..cw.min(jw) {
                    for k in 0..cpp {
                        v[(y * cw + x) * cpp + k] = px[(y * jw + x) * jc + k.min(jc - 1)] as u16;
                    }
                }
            }
            Ok(ChunkPx::U16(v))
        }
        comp::JPEG_XL => jxl_chunk(
            src,
            (cw, ch),
            ((info.width.saturating_sub(c.x) as usize).min(cw), (info.height.saturating_sub(c.y) as usize).min(ch)),
            cpp,
            bits,
            float,
            n,
        ),
        comp::ADOBE_DEFLATE | comp::DEFLATE => {
            let bytes_per = if float { bits.div_ceil(8) as usize } else { 0 };
            let row_bytes = if float { cw * cpp * bytes_per } else { (cw * cpp * bits as usize).div_ceil(8) };
            let raw = inflate(src, row_bytes * ch)?;
            unpack_chunk(&raw, order, packing, cw, ch, cpp, bits, float, info.predictor)
        }
        other => Err(RawError::Unsupported(format!("TIFF compression {other}"))),
    }
}

/// DNG JPEG XL tiles carry camera samples, not display RGB: preserve the codestream encoding
/// and integer precision instead of routing through the colour-managed standard-image codec.
fn jxl_chunk(src: &[u8], size: (usize, usize), minimum: (usize, usize), cpp: usize, bits: u32, float: bool, n: usize) -> Result<ChunkPx> {
    let (cw, ch) = size;
    let (min_w, min_h) = minimum;
    let err = |e| RawError::Corrupt(format!("JPEG XL tile: {e}"));
    let limit = n.saturating_mul(32).saturating_add(64 << 20).min(1 << 30);
    // Initialize incrementally so an input's dimensions are checked before feeding its frames.
    // Camera samples can exceed signed 16-bit range. Narrow Modular predictor buffers in
    // jxl-oxide can overflow on these tiles and then fail entropy validation; keep them 32-bit.
    let mut pending =
        jxl_oxide::JxlImage::builder().force_wide_buffers(true).alloc_tracker(jxl_oxide::AllocTracker::with_limit(limit)).build_uninit();
    let mut offset = 0usize;
    let mut image = loop {
        let end = offset.saturating_add(32).min(src.len());
        if end <= offset {
            return Err(RawError::Corrupt("Truncated JPEG XL header".into()));
        }
        let consumed = pending.feed_bytes(src.get(offset..end).ok_or_else(|| RawError::Corrupt("JPEG XL header bounds".into()))?).map_err(err)?;
        if consumed == 0 {
            return Err(RawError::Corrupt("JPEG XL header made no progress".into()));
        }
        offset = offset.saturating_add(consumed);
        match pending.try_init().map_err(err)? {
            jxl_oxide::InitializeResult::NeedMoreData(next) => pending = next,
            jxl_oxide::InitializeResult::Initialized(image) => break image,
        }
    };
    let header = image.image_header();
    if header.metadata.orientation != 1 || !(min_w..=cw).contains(&(image.width() as usize)) || !(min_h..=ch).contains(&(image.height() as usize)) {
        return Err(RawError::Corrupt("JPEG XL tile geometry does not match its TIFF layout".into()));
    }
    if header.metadata.bit_depth.bits_per_sample() != bits
        || matches!(header.metadata.bit_depth, jxl_oxide::image::BitDepth::FloatSample { .. }) != float
    {
        return Err(RawError::Corrupt("JPEG XL tile bit depth does not match its TIFF layout".into()));
    }
    let channels = if header.metadata.grayscale() { 1usize } else { 3usize };
    if channels.saturating_add(header.metadata.ec_info.len()) != cpp {
        return Err(RawError::Corrupt("JPEG XL tile channels do not match its TIFF layout".into()));
    }
    let camera_space = jxl_camera_space(header.metadata.xyb_encoded, &header.metadata.colour_encoding)?;
    if camera_space.is_some() {
        // XYB reconstruction normally gamut-maps intermediate sRGB before converting to
        // the declared space. Sensor values need the unclipped inverse, not display intent.
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(jxl_oxide::RenderingIntent::Perceptual));
    }
    while offset < src.len() {
        let consumed = image.feed_bytes(src.get(offset..).ok_or_else(|| RawError::Corrupt("JPEG XL frame bounds".into()))?).map_err(err)?;
        if consumed == 0 {
            return Err(RawError::Corrupt("JPEG XL frame made no progress".into()));
        }
        offset = offset.saturating_add(consumed);
    }
    let rendered = image.render_frame(0).map_err(err)?;
    let pixels = rendered.image_all_channels();
    if pixels.channels() != cpp {
        return Err(RawError::Corrupt("JPEG XL tile channels do not match its TIFF layout".into()));
    }
    let (fw, fh) = (pixels.width(), pixels.height());
    let mut values = vec![0.0; n];
    for y in 0..fh {
        for x in 0..fw {
            for c in 0..cpp {
                let input = (y * fw + x) * cpp + c;
                let output = (y * cw + x) * cpp + c;
                if let (Some(dst), Some(&value)) = (values.get_mut(output), pixels.buf().get(input)) {
                    if !value.is_finite() {
                        return Err(RawError::Corrupt("Non-finite JPEG XL sample".into()));
                    }
                    *dst = value;
                }
            }
        }
    }
    if let Some(matrix) = camera_space {
        for pixel in values.chunks_exact_mut(cpp) {
            if let [r, g, b, ..] = pixel {
                let rgb = matrix.apply_f32([*r, *g, *b]);
                if rgb.iter().any(|v| !v.is_finite()) {
                    return Err(RawError::Corrupt("Non-finite JPEG XL camera sample".into()));
                }
                [*r, *g, *b] = rgb;
            }
        }
    }
    Ok(if float {
        ChunkPx::F32(values)
    } else {
        let maximum = ((1u32 << bits.min(16)) - 1) as f32;
        ChunkPx::U16(values.into_iter().map(|v| (v * maximum).round().clamp(0.0, maximum) as u16).collect())
    })
}

/// Restore linear XYB tile samples with the existing colour library, without perceptual
/// gamut mapping. A non-linear/ICC encoding needs more than a matrix and is not inferred.
fn jxl_camera_space(xyb: bool, colour_encoding: &jxl_oxide::color::ColourEncoding) -> Result<Option<lightcraft_color::Mat3>> {
    use jxl_oxide::color::{ColourEncoding, ColourSpace, Primaries, TransferFunction, WhitePoint};
    use lightcraft_color::{D65, DISPLAY_P3, REC2020, RgbSpace, SRGB, Xy};
    if !xyb {
        return Ok(None);
    }
    let ColourEncoding::Enum(encoding) = colour_encoding else {
        return Err(RawError::Unsupported("ICC-encoded XYB DNG tile".into()));
    };
    if encoding.colour_space != ColourSpace::Rgb || encoding.tf != TransferFunction::Linear {
        return Err(RawError::Unsupported("non-linear RGB XYB DNG tile".into()));
    }
    let xy = |point: jxl_oxide::color::Customxy| Xy::new(point.x as f64 / 1e6, point.y as f64 / 1e6);
    let white = match encoding.white_point {
        WhitePoint::D65 => D65,
        WhitePoint::E => Xy::new(1.0 / 3.0, 1.0 / 3.0),
        WhitePoint::Dci => Xy::new(0.314, 0.351),
        WhitePoint::Custom(point) => xy(point),
    };
    let (r, g, b) = match encoding.primaries {
        Primaries::Srgb => (SRGB.r, SRGB.g, SRGB.b),
        Primaries::Bt2100 => (REC2020.r, REC2020.g, REC2020.b),
        Primaries::P3 => (DISPLAY_P3.r, DISPLAY_P3.g, DISPLAY_P3.b),
        Primaries::Custom { red, green, blue } => (xy(red), xy(green), xy(blue)),
    };
    if [r, g, b, white].iter().any(|p| !p.x.is_finite() || !p.y.is_finite() || p.y <= 0.0) {
        return Err(RawError::Corrupt("Invalid JPEG XL camera chromaticities".into()));
    }
    let space = RgbSpace { name: "JPEG XL camera samples", r, g, b, white };
    if space.to_xyz().inverse().is_none() {
        return Err(RawError::Corrupt("Degenerate JPEG XL camera primaries".into()));
    }
    Ok(Some(SRGB.to_space(&space)))
}

/// Decode one baseline JPEG chunk to 8-bit samples: (samples, width, height, channels).
fn lossy_jpeg(src: &[u8], cpp: usize) -> Result<(Vec<u8>, usize, usize, usize)> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let cs = if cpp == 1 { ColorSpace::Luma } else { ColorSpace::RGB };
    let opts = DecoderOptions::default().set_max_width(1 << 16).set_max_height(1 << 16).set_strict_mode(false).jpeg_set_out_colorspace(cs);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(src), opts);
    let px = d.decode().map_err(|e| RawError::Corrupt(format!("lossy JPEG tile: {e}")))?;
    let info = d.info().ok_or_else(|| RawError::Corrupt("lossy JPEG tile without a header".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let c = if cpp == 1 { 1 } else { 3 };
    if w == 0 || h == 0 || px.len() < w * h * c {
        return Err(RawError::Corrupt("lossy JPEG tile: short output".into()));
    }
    Ok((px, w, h, c))
}

#[allow(clippy::too_many_arguments)]
fn unpack_chunk(
    src: &[u8],
    order: ByteOrder,
    packing: Packing,
    cw: usize,
    ch: usize,
    cpp: usize,
    bits: u32,
    float: bool,
    predictor: u16,
) -> Result<ChunkPx> {
    let row_n = cw * cpp;
    let (factor, fp) = match predictor {
        1 => (0, false),
        2 => (1, false),
        3 => (1, true),
        34892 => (2, false),
        34893 => (4, false),
        34894 => (2, true),
        34895 => (4, true),
        p => return Err(RawError::Unsupported(format!("predictor {p}"))),
    };
    if float {
        let bp = bits.div_ceil(8) as usize;
        let row_bytes = row_n * bp;
        let mut out = vec![0f32; row_n * ch];
        let mut rowbuf = vec![0u8; row_bytes];
        for y in 0..ch {
            let s = src.get(y * row_bytes..).unwrap_or(&[]);
            let len = s.len().min(row_bytes);
            rowbuf[..len].copy_from_slice(&s[..len]);
            rowbuf[len..].fill(0);
            let big = if fp {
                undo_float_predictor(&mut rowbuf, row_n, bp, cpp * factor);
                true
            } else {
                if factor > 0 {
                    return Err(RawError::Unsupported("integer predictor on float data".into()));
                }
                order == ByteOrder::Big
            };
            for (i, o) in out[y * row_n..(y + 1) * row_n].iter_mut().enumerate() {
                let b = &rowbuf[i * bp..(i + 1) * bp];
                *o = match (bp, big) {
                    (2, true) => f16_to_f32(u16::from_be_bytes([b[0], b[1]])),
                    (2, false) => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    (3, true) => f24_to_f32(u32::from_be_bytes([0, b[0], b[1], b[2]])),
                    (3, false) => f24_to_f32(u32::from_le_bytes([b[0], b[1], b[2], 0])),
                    (4, true) => f32::from_be_bytes([b[0], b[1], b[2], b[3]]),
                    _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                };
            }
        }
        return Ok(ChunkPx::F32(out));
    }
    if fp {
        return Err(RawError::Unsupported("floating-point predictor on integer data".into()));
    }
    let mut out = vec![0u16; row_n * ch];
    let (row_bytes, kind) = match (bits, packing) {
        (8, _) => (row_n, 0),
        (16, _) | (_, Packing::Word16) => (row_n * 2, 1),
        (_, Packing::Msb) => ((row_n * bits as usize).div_ceil(8), 2),
        (_, Packing::Lsb) => ((row_n * bits as usize).div_ceil(8), 3),
    };
    for y in 0..ch {
        let s = src.get(y * row_bytes..).unwrap_or(&[]);
        let s = &s[..s.len().min(row_bytes)];
        let row = &mut out[y * row_n..(y + 1) * row_n];
        match kind {
            0 => row.iter_mut().zip(s).for_each(|(o, &b)| *o = b as u16),
            1 => read_u16s(s, order, row),
            2 => unpack_msb(s, bits, row),
            _ => unpack_lsb(s, bits, row),
        }
        if factor > 0 {
            if bits == 8 {
                for i in cpp * factor..row.len() {
                    row[i] = (row[i] as u8).wrapping_add(row[i - cpp * factor] as u8) as u16;
                }
            } else {
                undo_diff_u16(row, cpp * factor);
            }
        }
    }
    Ok(ChunkPx::U16(out))
}

#[cfg(test)]
mod lossy_tests {
    use super::*;
    use lightcraft_tiff::image::Layout;

    const JXL_TILE: &[u8] = &[
        0, 0, 0, 12, 74, 88, 76, 32, 13, 10, 135, 10, 0, 0, 0, 20, 102, 116, 121, 112, 106, 120, 108, 32, 0, 0, 0, 0, 106, 120, 108, 32, 0, 0, 0, 9,
        106, 120, 108, 108, 10, 0, 0, 0, 229, 106, 120, 108, 99, 255, 10, 24, 112, 252, 64, 2, 8, 4, 1, 0, 64, 3, 75, 18, 197, 130, 5, 82, 208, 253,
        70, 108, 12, 106, 12, 98, 12, 14, 198, 96, 45, 198, 24, 21, 21, 10, 116, 209, 200, 24, 201, 24, 131, 65, 93, 32, 64, 0, 24, 100, 90, 0, 116,
        1, 162, 48, 197, 152, 234, 96, 135, 165, 53, 34, 196, 72, 177, 34, 133, 32, 118, 138, 20, 41, 82, 172, 72, 177, 98, 247, 56, 197, 138, 20,
        43, 82, 172, 88, 59, 76, 177, 34, 197, 138, 20, 43, 82, 228, 92, 125, 56, 245, 207, 222, 81, 152, 83, 183, 105, 80, 148, 145, 248, 244, 87,
        90, 36, 151, 69, 132, 121, 217, 159, 30, 14, 232, 212, 223, 162, 75, 10, 127, 65, 33, 34, 214, 148, 4, 9, 18, 36, 72, 144, 32, 85, 172, 67,
        130, 4, 9, 18, 161, 79, 18, 80, 80, 14, 126, 249, 199, 110, 18, 247, 21, 115, 198, 241, 30, 241, 110, 100, 142, 126, 47, 30, 123, 123, 4, 57,
        127, 117, 109, 27, 210, 139, 196, 81, 139, 241, 151, 151, 255, 30, 62, 60, 217, 98, 248, 240, 225, 195, 247, 175, 79, 44, 104, 96, 9, 48, 56,
        160, 65, 18, 0, 64, 0, 208, 1, 64, 0,
    ];

    /// A lossy-DNG-style image: two 16×8 tiles, each a baseline JPEG, the right one cut short
    /// by the image edge.
    #[test]
    fn lossy_jpeg_tiles_decode() {
        let (tw, th) = (16usize, 8usize);
        let tile = |shade: u8| {
            let px: Vec<u8> = (0..tw * th).flat_map(|i| [shade, (i % tw * 15) as u8, 200]).collect();
            let mut out = Vec::new();
            jpeg_encoder::Encoder::new(&mut out, 100).encode(&px, tw as u16, th as u16, jpeg_encoder::ColorType::Rgb).unwrap();
            out
        };
        let (a, b) = (tile(40), tile(220));
        let mut file = vec![0u8; 8];
        let oa = file.len() as u64;
        file.extend_from_slice(&a);
        let ob = file.len() as u64;
        file.extend_from_slice(&b);
        let info = ImageInfo {
            width: 24,
            height: 8,
            bits_per_sample: vec![8, 8, 8],
            samples_per_pixel: 3,
            compression: comp::LOSSY_JPEG,
            photometric: 34892,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 0,
            layout: Layout::Tiles { tile_width: tw as u32, tile_height: th as u32 },
            offsets: vec![oa, ob],
            byte_counts: vec![a.len() as u64, b.len() as u64],
        };
        let RawData::U16(v) = read_image(&file, &info, ByteOrder::Little, Packing::Msb).unwrap() else { panic!("integer samples") };
        assert_eq!(v.len(), 24 * 8 * 3);
        let px = |x: usize, y: usize| &v[(y * 24 + x) * 3..(y * 24 + x) * 3 + 3];
        assert!((px(2, 3)[0] as i32 - 40).abs() <= 3 && (px(2, 3)[2] as i32 - 200).abs() <= 3, "{:?}", px(2, 3));
        assert!((px(20, 3)[0] as i32 - 220).abs() <= 3, "second tile: {:?}", px(20, 3));
        assert!((px(10, 5)[1] as i32 - 150).abs() <= 6, "green ramp: {:?}", px(10, 5));
    }

    /// Original procedural RGB ramp, losslessly encoded with cjxl 0.11.1. Embedded as code so
    /// tests need neither a native encoder nor a media fixture. Values deliberately exceed 8 bits.
    #[test]
    fn jpeg_xl_preserves_camera_samples_and_rejects_mismatched_headers() {
        let ChunkPx::U16(samples) = jxl_chunk(JXL_TILE, (8, 4), (8, 4), 3, 16, false, 96).unwrap() else { panic!("integer tile") };
        let expected: Vec<u16> = (0..32).flat_map(|i| [1001 + i * 1301, 60123 - i * 997, 251 + i * 1113]).collect();
        assert_eq!(samples, expected);
        for (w, h, cpp, bits, float) in [(7, 4, 3, 16, false), (8, 3, 3, 16, false), (8, 4, 1, 16, false), (8, 4, 3, 8, false), (8, 4, 3, 16, true)] {
            assert!(jxl_chunk(JXL_TILE, (w, h), (w, h), cpp, bits, float, w * h * cpp).is_err());
        }
        assert!(jxl_chunk(&JXL_TILE[..12], (8, 4), (8, 4), 3, 16, false, 96).is_err());
    }

    #[test]
    fn jpeg_xl_failed_tile_rejects_the_whole_image() {
        let mut file = JXL_TILE.to_vec();
        file.extend_from_slice(&[0; 32]);
        let mut info = ImageInfo {
            width: 16,
            height: 4,
            bits_per_sample: vec![16; 3],
            samples_per_pixel: 3,
            compression: comp::JPEG_XL,
            photometric: 34892,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 16,
            layout: Layout::Tiles { tile_width: 8, tile_height: 4 },
            offsets: vec![0, JXL_TILE.len() as u64],
            byte_counts: vec![JXL_TILE.len() as u64, 32],
        };
        assert!(read_image(&file, &info, ByteOrder::Little, Packing::Msb).is_err());
        info.offsets.pop();
        info.byte_counts.pop();
        assert!(read_image(&file, &info, ByteOrder::Little, Packing::Msb).is_err());
    }

    /// Own 64×64 step/ramp scene: RGB(x,y) is 65535 for x<32, otherwise x*131+y*91.
    /// cjxl 0.11.1, VarDCT distance .01/effort 9, linear Rec.2020. The declared narrow
    /// Modular header is insufficient for this scene's coefficients; the default decoder fails.
    #[test]
    fn jpeg_xl_high_range_predictors_use_wide_buffers() {
        const TILE: &[u8] = &[
            255, 10, 79, 240, 147, 144, 71, 131, 0, 19, 8, 0, 164, 5, 255, 255, 114, 12, 0, 128, 10, 21, 216, 54, 236, 81, 88, 56, 94, 40, 160, 80,
            88, 194, 9, 170, 23, 175, 225, 39, 20, 6, 36, 6, 223, 47, 33, 97, 10, 0, 247, 139, 241, 238, 46, 222, 197, 24, 116, 231, 189, 113, 68, 0,
            174, 3, 145, 186, 59, 59, 38, 163, 95, 150, 244, 1, 176, 132, 107, 54, 51, 191, 73, 200, 116, 2, 244, 44, 234, 184, 207, 108, 220, 84,
            55, 179, 1, 11, 194, 236, 124, 143, 67, 145, 135, 42, 57, 180, 71, 133, 4, 127, 39, 37, 6, 124, 2, 73, 153, 16, 195, 224, 211, 241, 158,
            1, 4, 188, 41, 64, 224, 33, 177, 133, 14, 170, 20, 11, 137, 48, 237, 132, 154, 0, 129, 2, 102, 7, 8, 221, 4, 0, 82, 160, 64, 209, 130,
            251, 251, 210, 93, 6, 90, 13, 136, 214, 177, 165, 73, 150, 218, 223, 202, 191, 132, 9, 212, 117, 7, 5, 196, 191, 224, 0, 0, 17, 138, 150,
            100, 30, 205, 1, 34, 162, 163, 163, 163, 207, 177, 177, 81, 241, 250, 141, 149, 2, 176, 190, 55, 188, 141, 179, 241, 245, 132, 150, 126,
            115, 226, 177, 249, 145, 5, 22, 150, 187, 113, 45, 58, 249, 127, 27, 55, 112, 241, 46, 175, 60, 184, 46, 248, 213, 3, 93, 110, 168, 146,
            113, 11, 163, 171, 87, 66, 164, 185, 37, 0, 65, 162, 15, 113, 57, 205, 163, 146, 14, 253, 85, 132, 110, 3, 82, 62, 56, 22, 75, 53, 48, 2,
            176, 154, 22, 81, 154, 8, 20, 144, 198, 178, 140, 149, 70, 222, 32, 221, 193, 25, 223, 202, 76, 155, 159, 44, 93, 242, 245, 76, 195, 79,
            121, 122, 162, 208, 55, 43, 10, 199, 145, 252, 39, 9, 97, 147, 34, 16, 135, 16, 243, 215, 100, 90, 210, 230, 113, 11, 69, 240, 213, 146,
            159, 79, 49, 80, 108, 44, 235, 145, 195, 7, 163, 238, 129, 217, 203, 66, 64, 22, 195, 46, 136, 15, 23, 42, 54, 41, 94, 5, 64, 205, 111,
            127, 70, 14, 232, 7,
        ];
        let narrow = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(TILE)).unwrap();
        assert!(narrow.render_frame(0).is_err());
        let ChunkPx::U16(samples) = jxl_chunk(TILE, (64, 64), (64, 64), 3, 16, false, 64 * 64 * 3).unwrap() else { panic!("integer tile") };
        // Native djxl reference samples of this original procedural scene, within float roundoff.
        for (pixel, expected) in [(0, 65535), (32, 4198), (33, 4324), (63, 8251), (100, 4805), (1000, 6605), (4095, 13986)] {
            for channel in 0..3 {
                assert!((samples[pixel * 3 + channel] as i32 - expected).abs() <= 2);
            }
        }
    }

    /// Own saturated cyan scene (6000,64000,62000), encoded as linear Rec.2020 XYB.
    /// Intermediate sRGB is outside its gamut; display gamut mapping would destroy camera data.
    #[test]
    fn jpeg_xl_xyb_reconstructs_camera_samples_without_gamut_mapping() {
        const TILE: &[u8] = &[
            255, 10, 24, 112, 252, 36, 228, 209, 32, 0, 19, 8, 0, 208, 0, 255, 255, 114, 12, 0, 128, 10, 149, 81, 198, 13, 94, 206, 245, 124, 249,
            161, 195, 98, 218, 198, 117, 134, 182, 218, 182, 176, 132, 49, 241, 140, 131, 133, 132, 20, 176, 113, 113, 96, 216, 205, 41, 60, 118,
            217, 244, 199, 64, 208, 162, 18, 0,
        ];
        let ChunkPx::U16(samples) = jxl_chunk(TILE, (8, 4), (8, 4), 3, 16, false, 96).unwrap() else { panic!("integer tile") };
        for pixel in samples.chunks_exact(3) {
            for (&actual, expected) in pixel.iter().zip([6001, 63994, 62001]) {
                assert!((actual as i32 - expected).abs() <= 2, "{pixel:?}");
            }
        }
        let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(TILE)).unwrap();
        let mut color = image.image_header().metadata.colour_encoding.clone();
        let jxl_oxide::color::ColourEncoding::Enum(encoding) = &mut color else { panic!("enum encoding") };
        encoding.tf = jxl_oxide::color::TransferFunction::Srgb;
        assert!(jxl_camera_space(true, &color).is_err());
        let jxl_oxide::color::ColourEncoding::Enum(encoding) = &mut color else { panic!("enum encoding") };
        encoding.tf = jxl_oxide::color::TransferFunction::Linear;
        encoding.white_point = jxl_oxide::color::WhitePoint::Custom(jxl_oxide::color::Customxy { x: 312700, y: 0 });
        assert!(jxl_camera_space(true, &color).is_err());
        color = jxl_oxide::color::ColourEncoding::IccProfile(jxl_oxide::color::ColourSpace::Rgb);
        assert!(jxl_camera_space(true, &color).is_err());
    }

    /// Own constant float scene: linear values include highlight headroom above 1.0.
    #[test]
    fn jpeg_xl_preserves_floating_point_camera_samples() {
        const TILE: &[u8] = &[
            0, 0, 0, 12, 74, 88, 76, 32, 13, 10, 135, 10, 0, 0, 0, 20, 102, 116, 121, 112, 106, 120, 108, 32, 0, 0, 0, 0, 106, 120, 108, 32, 0, 0, 0,
            9, 106, 120, 108, 108, 10, 0, 0, 0, 79, 106, 120, 108, 99, 255, 10, 24, 112, 114, 144, 8, 4, 1, 0, 236, 0, 75, 18, 197, 130, 133, 36,
            218, 142, 184, 177, 133, 144, 5, 155, 40, 150, 252, 9, 0, 0, 0, 47, 0, 0, 4, 30, 0, 84, 5, 0, 0, 4, 0, 0, 176, 7, 0, 64, 0, 7, 64, 42, 0,
            0, 0, 1, 0, 0, 252, 2, 0, 16, 104, 0, 170, 2, 0, 0, 0,
        ];
        let ChunkPx::F32(samples) = jxl_chunk(TILE, (8, 4), (8, 4), 3, 32, true, 96).unwrap() else { panic!("float tile") };
        assert_eq!(samples, [0.125, 0.375, 1.5].repeat(32));
    }
}
