//! Nikon NEF / NRW — uncompressed and Huffman-compressed (lossless, lossy) variants.
//!
//! Sources: TIFF 6.0 (the raw image is a standard CFA SubIFD), Laurent Clévy's NEF structure notes (prose: IFD
//! layout, SubIFDs, maker note header) and the ExifTool Nikon tag-name documentation (`0x000c` WB_RBLevels,
//! `0x003d` BlackLevel, `0x0045` CropArea, `0x0096` NEFLinearizationTable). The Huffman-compressed data (compression 34713) is decoded
//! by [`super::nefc`], which documents its clean-room sources; files it can't decode yet ("lossy after split")
//! are reported as [`RawError::Unsupported`] and their embedded previews still work.

use super::{nefc, white_from_data};
use crate::tiffraw::{Packing, read_image};
use crate::{BlackLevel, Cfa, ColorData, OpcodeLists, RawData, RawError, RawFormat, RawImage, Rect, Result};
use lightcraft_geom::Orientation;
use lightcraft_tiff::image::ImageInfo;
use lightcraft_tiff::tags::{self as t, photometric};
use lightcraft_tiff::{ByteOrder, Ifd, Tiff, Value, makernote};

const WB_RB_LEVELS: u16 = 0x000c;
const BLACK_LEVEL: u16 = 0x003d;
const CROP_AREA: u16 = 0x0045;
const LINEARIZATION_TABLE: u16 = 0x0096;

fn raw_ifd(tiff: &Tiff) -> Option<&Ifd> {
    tiff.all_ifds()
        .into_iter()
        .filter(|i| i.u16(t::PHOTOMETRIC) == Some(photometric::CFA))
        .max_by_key(|i| i.u64(t::IMAGE_WIDTH).unwrap_or(0).saturating_mul(i.u64(t::IMAGE_LENGTH).unwrap_or(0)))
}

/// Modern Nikon maker-note CropArea is unsigned SHORT [left, top, width, height], in stored sensor coordinates.
/// See https://exiftool.org/TagNames/Nikon.html. Keep the active plane intact so black/CFA phase stays anchored
/// to its original origin; `RawImage` applies this default crop after demosaicing or when selecting bin blocks.
fn declared_crop(note: &[u8], mn: &makernote::MakerNote, w: usize, h: usize) -> Option<Rect> {
    if !note.starts_with(b"Nikon\0\x02") || mn.kind != makernote::MakerNoteKind::NikonV3 {
        return None;
    }
    let Value::Short(values) = mn.ifd.value(CROP_AREA)? else { return None };
    let [left, top, width, height] = values.as_slice() else { return None };
    let (left, top, width, height) = (usize::from(*left), usize::from(*top), usize::from(*width), usize::from(*height));
    if width == 0 || height == 0 || left.checked_add(width)? > w || top.checked_add(height)? > h {
        return None;
    }
    Some(Rect::new(left, top, width, height))
}

/// Width without the optically masked columns some bodies append on the right: trailing columns (at most 64)
/// whose mean is below 1% of the white level while the image interior is brighter. Kept even for CFA phase.
pub(crate) fn trailing_masked_columns(d: &[u16], w: usize, h: usize, white: f32) -> usize {
    if w < 128 || h == 0 {
        return w;
    }
    let step = (h / 256).max(1);
    let col_mean = |x: usize| (0..h).step_by(step).map(|y| d[y * w + x] as f64).sum::<f64>() / h.div_ceil(step) as f64;
    let interior = (w / 4..w * 3 / 4).step_by(w / 64).map(col_mean).sum::<f64>() / 32.0;
    let dark = (white as f64 * 0.01).min(interior * 0.1);
    let mut aw = w;
    while aw > w - 64 && col_mean(aw - 1) < dark {
        aw -= 1;
    }
    aw & !1
}

pub(crate) fn decode(bytes: &[u8]) -> Result<RawImage> {
    let tiff = Tiff::parse(bytes)?;
    let ifd0 = &tiff.ifds[0];
    let raw = raw_ifd(&tiff).ok_or_else(|| RawError::Unsupported("NEF without a CFA image IFD".into()))?;
    let info = raw.image()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let bits = info.bits() as u32;
    let make = ifd0.string(t::MAKE).unwrap_or_default();
    let note = tiff.exif().and_then(|e| e.get(t::MAKER_NOTE));
    let mn = note.and_then(|e| makernote::parse_makernote(bytes, e.offset, e.count() as u64, tiff.order, &make));
    let data =
        if info.compression == t::compression::NIKON { compressed(bytes, &info, mn.as_ref())? } else { uncompressed(bytes, &info, tiff.order)? };
    let RawData::U16(ref samples) = data else { return Err(RawError::Unsupported("float NEF".into())) };

    // Maker note 0x003d is in 14-bit units whatever the sample depth: 12-bit files from the D750, D780, D850,
    // D7500 and Z 50 store 600 / 1008 / 400 / 400 / 1008 while their darkest samples are 150 / ~252 / 99 / 99 / ~250
    // (our measurements on CC0 raw.pixls.us samples; the same bodies' 14-bit files match the tag as is).
    let black_scale = if bits == 12 { 0.25 } else { 1.0 };
    let black = match mn.as_ref().and_then(|m| m.ifd.f64s(BLACK_LEVEL)).as_deref() {
        Some([a, b, c, d]) if [a, b, c, d].iter().all(|v| **v < 16384.0) => BlackLevel {
            repeat_rows: 2,
            repeat_cols: 2,
            values: [a, b, c, d].iter().map(|v| (**v * black_scale) as f32).collect(),
            ..Default::default()
        },
        _ => BlackLevel::uniform(0.0),
    };
    let wb = mn
        .as_ref()
        .and_then(|m| m.ifd.f64s(WB_RB_LEVELS))
        .filter(|v| v.len() >= 2 && v[0] > 0.1 && v[1] > 0.1 && v[0] < 10.0 && v[1] < 10.0)
        .map(|v| [v[0] as f32, 1.0, v[1] as f32]);
    let cfa = match (raw.u64s(t::CFA_REPEAT_PATTERN_DIM).as_deref(), raw.bytes(t::CFA_PATTERN_EP)) {
        (Some([2, 2]), Some(p)) if p.len() == 4 && p.iter().all(|&c| c <= 2) => Cfa { width: 2, height: 2, pattern: p.to_vec() },
        _ => Cfa::bayer_static("RGGB"),
    };
    let white = white_from_data(samples, bits);
    let crop = note.and_then(|n| n.value.as_bytes()).zip(mn.as_ref()).and_then(|(n, mn)| declared_crop(n, mn, w, h));
    let (active_area, crop) = if let Some(crop) = crop {
        (Rect::new(0, 0, w, h), crop)
    } else {
        let active_w = trailing_masked_columns(samples, w, h, white);
        let active = Rect::new(0, 0, active_w, h);
        (active, active)
    };
    let mut metadata = lightcraft_meta::from_tiff(&tiff);
    metadata.width = Some(crop.width as u32);
    metadata.height = Some(crop.height as u32);
    let img = RawImage {
        format: RawFormat::Nef,
        width: w,
        height: h,
        cpp: 1,
        data,
        cfa: Some(cfa),
        bits,
        black,
        white: vec![white],
        active_area,
        crop,
        orientation: Orientation::from_exif(ifd0.u16(t::ORIENTATION).unwrap_or(1)),
        color: ColorData::default(),
        wb_multipliers: wb,
        linearized: false,
        opcodes: OpcodeLists::default(),
        metadata,
    };
    img.validate()?;
    Ok(img)
}

/// Uncompressed strips: 16-bit words or 12-bit MSB-packed, told apart by the strip size.
fn uncompressed(bytes: &[u8], info: &ImageInfo, order: ByteOrder) -> Result<RawData> {
    let (w, h) = (info.width as usize, info.height as usize);
    let bits = info.bits() as u32;
    let row_samples = w * info.samples_per_pixel as usize;
    let chunks = info.chunks(bytes.len() as u64);
    let rows_in_first = chunks.first().map(|c| c.height as usize).unwrap_or(h).max(1);
    let bytes_per_row = chunks.first().map(|c| c.len as usize / rows_in_first).unwrap_or(0);
    let packing = if bytes_per_row >= row_samples * 2 {
        Packing::Word16
    } else if bits == 12 && bytes_per_row * 8 >= row_samples * 12 && bytes_per_row * 8 < row_samples * 13 {
        Packing::Msb
    } else if info.compression == 1 && bits != 8 && bits != 16 {
        return Err(RawError::Unsupported(format!("NEF uncompressed packing ({bytes_per_row} bytes per {row_samples}-sample row)")));
    } else {
        Packing::Msb
    };
    read_image(bytes, info, order, packing)
}

/// Nikon Huffman-compressed strip (see [`super::nefc`]); the decode table is maker note `0x0096`.
fn compressed(bytes: &[u8], info: &ImageInfo, mn: Option<&makernote::MakerNote>) -> Result<RawData> {
    // (some Z bodies label uncompressed, row-padded data 34713 without a table: not handled here yet)
    let table = mn
        .and_then(|m| Some((m.ifd.bytes(LINEARIZATION_TABLE)?, m.order)))
        .ok_or_else(|| RawError::Unsupported("Nikon compressed NEF without a linearization table (maker note 0x96)".into()))?;
    if info.samples_per_pixel != 1 {
        return Err(RawError::Unsupported(format!("Nikon compressed NEF with {} samples per pixel", info.samples_per_pixel)));
    }
    let bits = info.bits() as u32;
    let table = nefc::parse_table(table.0, table.1, bits)?;
    // one strip in every sample; several would be contiguous parts of the same bit stream
    let chunks = info.chunks(bytes.len() as u64);
    let (Some(first), Some(last)) = (chunks.first(), chunks.last()) else { return Err(RawError::Corrupt("NEF: no image data".into())) };
    let end = last.offset.saturating_add(last.len).min(bytes.len() as u64);
    let src = usize::try_from(first.offset).ok().zip(usize::try_from(end).ok()).and_then(|(a, b)| bytes.get(a..b));
    let src = src.ok_or_else(|| RawError::Corrupt("NEF: image data outside the file".into()))?;
    Ok(RawData::U16(nefc::decode(src, info.width as usize, info.height as usize, bits, &table)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_tiff::{ByteOrder, IfdBuilder, ImageData, TiffWriter, Value};

    fn nef(compression: u16, bits: u16, strips: Vec<Vec<u8>>, w: u32, h: u32, rps: u32) -> Vec<u8> {
        nef_with_note(compression, bits, strips, w, h, rps, None)
    }

    /// [`nef`] with an optional maker note (`Nikon\0` v2 header + embedded big-endian TIFF holding `note`).
    fn nef_with_note(compression: u16, bits: u16, strips: Vec<Vec<u8>>, w: u32, h: u32, rps: u32, note: Option<IfdBuilder>) -> Vec<u8> {
        let mut raw = IfdBuilder::new();
        raw.set(t::NEW_SUBFILE_TYPE, Value::Long(vec![0]));
        raw.set(t::IMAGE_WIDTH, Value::Long(vec![w]));
        raw.set(t::IMAGE_LENGTH, Value::Long(vec![h]));
        raw.set(t::BITS_PER_SAMPLE, Value::Short(vec![bits]));
        raw.set(t::COMPRESSION, Value::Short(vec![compression]));
        raw.set(t::PHOTOMETRIC, Value::Short(vec![photometric::CFA]));
        raw.set(t::CFA_REPEAT_PATTERN_DIM, Value::Short(vec![2, 2]));
        raw.set(t::CFA_PATTERN_EP, Value::Byte(vec![2, 1, 1, 0]));
        raw.set_image(ImageData::Strips { rows_per_strip: rps, strips });
        let mut ifd0 = IfdBuilder::new();
        ifd0.set(t::MAKE, Value::Ascii("NIKON CORPORATION".into()));
        ifd0.set(t::MODEL, Value::Ascii("NIKON TEST".into()));
        if let Some(mn) = note {
            let mut bytes = b"Nikon\0\x02\x10\0\0".to_vec();
            bytes.extend(TiffWriter::new(ByteOrder::Big, false).write(&[mn]).unwrap());
            let mut exif = IfdBuilder::new();
            exif.set(t::MAKER_NOTE, Value::Undefined(bytes));
            ifd0.set_child(t::EXIF_IFD, exif);
        }
        ifd0.add_sub_ifd(raw);
        TiffWriter::new(ByteOrder::Big, false).write(&[ifd0]).unwrap()
    }

    #[test]
    fn black_level_tag_is_in_14_bit_units() {
        let (w, h) = (16usize, 4usize);
        let words: Vec<u8> = (0..w * h).flat_map(|i| (100 + i as u16).to_be_bytes()).collect();
        let note = || {
            let mut mn = IfdBuilder::new();
            mn.set(BLACK_LEVEL, Value::Short(vec![400, 404, 408, 412]));
            mn
        };
        let black = |bits| crate::decode(&nef_with_note(1, bits, vec![words.clone()], w as u32, h as u32, h as u32, Some(note()))).unwrap().black;
        assert_eq!(black(12).values, vec![100.0, 101.0, 102.0, 103.0]);
        assert_eq!(black(14).values, vec![400.0, 404.0, 408.0, 412.0]);
    }

    #[test]
    fn declared_crop_preserves_sensor_and_cfa_phase_for_full_and_binned_development() {
        let (w, h) = (32usize, 24usize);
        let cfa = Cfa::bayer_static("BGGR");
        let black = [100u16, 101, 102, 103];
        let levels = [800u16, 400, 200];
        let samples: Vec<u16> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                black[(y % 2) * 2 + x % 2] + levels[cfa.color_at(x, y) as usize]
            })
            .collect();
        let words: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
        // Arbitrary even and odd offsets: the crop must not shift the Bayer or per-position black phase.
        for crop in [Rect::new(2, 4, 18, 12), Rect::new(3, 5, 20, 14)] {
            let mut note = IfdBuilder::new();
            note.set(CROP_AREA, Value::Short(vec![crop.x as u16, crop.y as u16, crop.width as u16, crop.height as u16]));
            note.set(BLACK_LEVEL, Value::Short(black.map(|v| v * 4).to_vec()));
            let r = crate::decode(&nef_with_note(1, 12, vec![words.clone()], w as u32, h as u32, h as u32, Some(note))).unwrap();
            assert_eq!((r.width, r.height), (w, h));
            assert_eq!(r.data, RawData::U16(samples.clone()));
            assert_eq!(r.active_area, Rect::new(0, 0, w, h));
            assert_eq!(r.crop, crop);
            assert_eq!(r.cfa.as_ref().unwrap(), &cfa);
            assert_eq!(r.black.values, black.map(f32::from));
            assert_eq!((r.metadata.width, r.metadata.height), (Some(crop.width as u32), Some(crop.height as u32)));
            assert_eq!(r.info().developed_size(), (crop.width, crop.height));
            let full = r.develop(crate::Method::Bilinear).unwrap();
            let binned = r.develop_binned(2, 0.99).unwrap().unwrap();
            assert_eq!((full.width, full.height), (crop.width, crop.height));
            assert_eq!((binned.width, binned.height), (crop.width / 2, crop.height / 2));
            let expected = levels.map(|v| f32::from(v) / (r.white_at(0) - r.black.mean()));
            for px in full.data.iter().chain(&binned.data) {
                for (value, expected) in px.iter().zip(expected) {
                    assert!((*value - expected).abs() < 1e-6, "{px:?} != {expected:?}");
                }
            }
        }
    }

    #[test]
    fn declared_crop_overrides_trailing_dark_column_guess() {
        let (w, h) = (160usize, 20usize);
        let samples: Vec<u16> = (0..w * h).map(|i| if i % w < w - 8 { 500 } else { 0 }).collect();
        let words: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
        let legacy = crate::decode(&nef(1, 12, vec![words.clone()], w as u32, h as u32, h as u32)).unwrap();
        assert_eq!(legacy.active_area, Rect::new(0, 0, w - 8, h));
        let mut note = IfdBuilder::new();
        note.set(CROP_AREA, Value::Short(vec![2, 2, 158, 16]));
        let r = crate::decode(&nef_with_note(1, 12, vec![words], w as u32, h as u32, h as u32, Some(note))).unwrap();
        assert_eq!(r.active_area, Rect::new(0, 0, w, h));
        assert_eq!(r.crop, Rect::new(2, 2, 158, 16));
        assert_eq!(r.data, legacy.data);
        assert_eq!(r.cfa, legacy.cfa);
    }

    #[test]
    fn declared_crop_selects_the_exact_sensor_region() {
        let (w, h) = (40usize, 30usize);
        let cfa = Cfa::bayer_static("BGGR");
        let samples: Vec<u16> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let rgb = [400 + 10 * x + 4 * y, 700 + 2 * x + 5 * y, 900 + 3 * x + 6 * y];
                rgb[cfa.color_at(x, y) as usize] as u16
            })
            .collect();
        let words: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
        let baseline = crate::decode(&nef(1, 12, vec![words.clone()], w as u32, h as u32, h as u32)).unwrap();
        let crop = Rect::new(5, 3, 28, 22);
        let mut note = IfdBuilder::new();
        note.set(CROP_AREA, Value::Short(vec![5, 3, 28, 22]));
        let r = crate::decode(&nef_with_note(1, 12, vec![words], w as u32, h as u32, h as u32, Some(note))).unwrap();
        let sensor_rgb = baseline.develop(crate::Method::Bilinear).unwrap();
        let cropped_rgb = r.develop(crate::Method::Bilinear).unwrap();
        for y in 0..crop.height {
            for x in 0..crop.width {
                assert_eq!(cropped_rgb.get(x, y), sensor_rgb.get(crop.x + x, crop.y + y));
            }
        }
        let binned = r.develop_binned(2, 0.99).unwrap().unwrap();
        for by in 0..binned.height {
            for bx in 0..binned.width {
                let (mut sum, mut count) = ([0f32; 3], [0u32; 3]);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (x, y) = (crop.x + 2 * bx + dx, crop.y + 2 * by + dy);
                        let channel = cfa.color_at(x, y) as usize;
                        sum[channel] += f32::from(samples[y * w + x]) / r.white_at(0);
                        count[channel] += 1;
                    }
                }
                for (channel, value) in binned.get(bx, by).iter().enumerate() {
                    assert!((*value - sum[channel] / count[channel] as f32).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn malformed_or_unknown_dialect_crop_preserves_legacy_geometry() {
        let (w, h) = (160usize, 20usize);
        let words: Vec<u8> = (0..w * h).flat_map(|i| if i % w < w - 8 { 500u16.to_be_bytes() } else { 0u16.to_be_bytes() }).collect();
        let baseline = crate::decode(&nef(1, 12, vec![words.clone()], w as u32, h as u32, h as u32)).unwrap();
        let invalid = [
            Value::Short(vec![2, 2, 154]),
            Value::Short(vec![2, 2, 154, 16, 0]),
            Value::Short(vec![2, 2, 0, 16]),
            Value::Short(vec![2, 2, 154, 0]),
            Value::Short(vec![161, 2, 1, 16]),
            Value::Short(vec![2, 21, 154, 1]),
            Value::Short(vec![2, 2, 159, 16]),
            Value::Short(vec![2, 2, 154, 19]),
            Value::Short(vec![u16::MAX, u16::MAX, u16::MAX, u16::MAX]),
            Value::Long(vec![2, 2, 154, 16]),
            Value::SShort(vec![2, 2, 154, 16]),
            Value::Float(vec![2.0, 2.0, 154.0, 16.0]),
            Value::Undefined(vec![2, 2, 154, 16]),
        ];
        for value in invalid {
            let mut note = IfdBuilder::new();
            note.set(CROP_AREA, value);
            let r = crate::decode(&nef_with_note(1, 12, vec![words.clone()], w as u32, h as u32, h as u32, Some(note))).unwrap();
            assert_eq!(r, baseline);
        }
        // A Nikon-looking unknown version can still parse as a modern maker note; it must not authorize CropArea.
        for version in [0, 1, 3] {
            let mut note = IfdBuilder::new();
            note.set(CROP_AREA, Value::Short(vec![2, 2, 154, 16]));
            let mut bytes = nef_with_note(1, 12, vec![words.clone()], w as u32, h as u32, h as u32, Some(note));
            let start = bytes.windows(7).position(|v| v == b"Nikon\0\x02").unwrap();
            bytes[start + 6] = version;
            let r = crate::decode(&bytes).unwrap();
            assert_eq!((r.active_area, r.crop), (baseline.active_area, baseline.crop));
            assert_eq!(r.data, baseline.data);
        }
    }

    #[test]
    fn uncompressed_word16_and_packed12() {
        let (w, h) = (16usize, 6usize);
        let px: Vec<u16> = (0..w * h).map(|i| (i * 131 % 4096) as u16).collect();
        let words: Vec<u8> = px.iter().flat_map(|v| v.to_be_bytes()).collect();
        let bytes = nef(1, 12, words.chunks(w * 2 * 3).map(|c| c.to_vec()).collect(), w as u32, h as u32, 3);
        assert_eq!(crate::probe(&bytes), Some(RawFormat::Nef));
        let r = crate::decode(&bytes).unwrap();
        assert_eq!(r.data, RawData::U16(px.clone()));
        assert_eq!(r.cfa.as_ref().unwrap().name(), "BGGR");
        // 12-bit MSB packed
        let mut packed = Vec::new();
        for pair in px.chunks(2) {
            let (a, b) = (pair[0] as u32, pair[1] as u32);
            let v = (a << 12) | b;
            packed.extend_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]);
        }
        let bytes = nef(1, 12, vec![packed], w as u32, h as u32, h as u32);
        assert_eq!(crate::decode(&bytes).unwrap().data, RawData::U16(px));
    }

    #[test]
    fn compressed_without_table_is_unsupported() {
        let bytes = nef(34713, 14, vec![vec![0; 64]], 8, 8, 8);
        assert!(matches!(crate::decode(&bytes), Err(RawError::Unsupported(_))));
    }
}
