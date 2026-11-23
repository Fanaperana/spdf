//! Pure geometry for the neural OCR pipeline: DBNet probability-map
//! post-processing (connected components → minimum-area rotated rectangles →
//! unclip), and rotated-rectangle crop sampling for the recognizer.
//!
//! Everything here is backend-free and unit-tested on synthetic inputs.

/// A rotated rectangle. `angle` is the direction of the long ("reading")
/// axis in radians, image coordinates (x right, y down), normalised to
/// `(-π/2, π/2]`. `w` is measured along that axis, `h` across it, `w >= h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotRect {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub angle: f64,
}

impl RotRect {
    /// Unit vector along the reading axis.
    pub fn u(&self) -> (f64, f64) {
        (self.angle.cos(), self.angle.sin())
    }
    /// Unit vector across the reading axis (pointing "down" for upright text).
    pub fn v(&self) -> (f64, f64) {
        (-self.angle.sin(), self.angle.cos())
    }
    /// Point at reading-axis offset `s` and cross-axis offset `t` from the centre.
    pub fn at(&self, s: f64, t: f64) -> (f64, f64) {
        let (ux, uy) = self.u();
        let (vx, vy) = self.v();
        (self.cx + ux * s + vx * t, self.cy + uy * s + vy * t)
    }
    /// The four corners.
    pub fn corners(&self) -> [(f64, f64); 4] {
        let (hw, hh) = (self.w / 2.0, self.h / 2.0);
        [
            self.at(-hw, -hh),
            self.at(hw, -hh),
            self.at(hw, hh),
            self.at(-hw, hh),
        ]
    }
    /// Axis-aligned bounding box `[x1, y1, x2, y2]`.
    pub fn aabb(&self) -> [f64; 4] {
        let c = self.corners();
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (x, y) in c {
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        }
        b
    }
    /// True when the long axis is closer to vertical than horizontal.
    pub fn is_vertical(&self) -> bool {
        self.angle.abs() > std::f64::consts::FRAC_PI_4
    }
}

/// One detected text region on the detector's probability map.
#[derive(Debug, Clone)]
pub struct DetBox {
    /// Region grown by the unclip offset: what the recogniser crops.
    pub rect: RotRect,
    /// Region before unclipping: the tight text extent.
    pub core: RotRect,
    /// Mean probability over the region's pixels.
    pub score: f64,
}

/// DBNet post-processing parameters (PaddleOCR defaults).
#[derive(Debug, Clone, Copy)]
pub struct DbParams {
    pub thresh: f32,
    pub box_thresh: f64,
    pub unclip_ratio: f64,
    pub min_size: f64,
    pub max_candidates: usize,
}

impl Default for DbParams {
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
            min_size: 3.0,
            max_candidates: 3000,
        }
    }
}

/// Turn a `w × h` probability map into text boxes in map coordinates.
pub fn db_boxes(prob: &[f32], w: usize, h: usize, p: &DbParams) -> Vec<DetBox> {
    assert_eq!(prob.len(), w * h);
    let mut label = vec![0u32; w * h];
    let mut next = 0u32;
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for start in 0..w * h {
        if label[start] != 0 || prob[start] <= p.thresh {
            continue;
        }
        next += 1;
        label[start] = next;
        stack.push(start);
        pts.clear();
        let mut sum = 0.0f64;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            sum += prob[i] as f64;
            // Pixel corners, so a 1-pixel-wide stroke still has an area.
            pts.push((x as f64, y as f64));
            pts.push((x as f64 + 1.0, y as f64));
            pts.push((x as f64, y as f64 + 1.0));
            pts.push((x as f64 + 1.0, y as f64 + 1.0));
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let j = ny * w + nx;
                    if label[j] == 0 && prob[j] > p.thresh {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        let n = pts.len() / 4;
        let score = sum / n as f64;
        if score < p.box_thresh {
            continue;
        }
        let hull = convex_hull(&mut pts);
        let Some(rect) = min_area_rect(&hull) else {
            continue;
        };
        if rect.h < p.min_size {
            continue;
        }
        let core = rect;
        let rect = unclip(rect, p.unclip_ratio);
        if rect.h < p.min_size + 2.0 {
            continue;
        }
        out.push(DetBox { rect, core, score });
        if out.len() >= p.max_candidates {
            break;
        }
    }
    out
}

/// Grow a rectangle by the Vatti-clipping offset DBNet trains against:
/// `d = area · ratio / perimeter` on every side.
pub fn unclip(r: RotRect, ratio: f64) -> RotRect {
    let area = r.w * r.h;
    let perim = 2.0 * (r.w + r.h);
    if perim <= 0.0 {
        return r;
    }
    let d = area * ratio / perim;
    RotRect {
        w: r.w + 2.0 * d,
        h: r.h + 2.0 * d,
        ..r
    }
}

/// Andrew's monotone chain. Sorts and dedups `pts` in place; returns the hull
/// counter-clockwise without repeating the first point.
pub fn convex_hull(pts: &mut Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup();
    if pts.len() < 3 {
        return pts.clone();
    }
    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(pts.len() * 2);
    for &p in pts.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Minimum-area enclosing rectangle of a convex polygon (rotating calipers
/// over edge directions). Returns `None` for degenerate input.
pub fn min_area_rect(hull: &[(f64, f64)]) -> Option<RotRect> {
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, RotRect)> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (vx, vy) = (-uy, ux);
        let (mut s0, mut s1, mut t0, mut t1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &(x, y) in hull {
            let s = x * ux + y * uy;
            let t = x * vx + y * vy;
            s0 = s0.min(s);
            s1 = s1.max(s);
            t0 = t0.min(t);
            t1 = t1.max(t);
        }
        let area = (s1 - s0) * (t1 - t0);
        if best.as_ref().is_none_or(|(a, _)| area < *a - 1e-9) {
            let (sc, tc) = ((s0 + s1) / 2.0, (t0 + t1) / 2.0);
            let rect = RotRect {
                cx: ux * sc + vx * tc,
                cy: uy * sc + vy * tc,
                w: s1 - s0,
                h: t1 - t0,
                angle: uy.atan2(ux),
            };
            best = Some((area, rect));
        }
    }
    best.map(|(_, r)| normalize(r))
}

/// Make `w` the long side and fold the angle into `(-π/2, π/2]`.
pub fn normalize(mut r: RotRect) -> RotRect {
    use std::f64::consts::{FRAC_PI_2, PI};
    if r.h > r.w {
        std::mem::swap(&mut r.w, &mut r.h);
        r.angle += FRAC_PI_2;
    }
    while r.angle > FRAC_PI_2 + 1e-12 {
        r.angle -= PI;
    }
    while r.angle <= -FRAC_PI_2 + 1e-12 {
        r.angle += PI;
    }
    r
}

/// Interleaved RGB8 image view.
pub struct Rgb<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
}

impl Rgb<'_> {
    /// Bilinear sample, white outside the image.
    fn sample(&self, x: f64, y: f64, out: &mut [f32; 3]) {
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = (x - x0) as f32;
        let fy = (y - y0) as f32;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let mut acc = [0f32; 3];
        for (dy, wy) in [(0i64, 1.0 - fy), (1, fy)] {
            for (dx, wx) in [(0i64, 1.0 - fx), (1, fx)] {
                let wgt = wx * wy;
                if wgt == 0.0 {
                    continue;
                }
                let (px, py) = (x0 + dx, y0 + dy);
                if px < 0 || py < 0 || px >= self.w as i64 || py >= self.h as i64 {
                    for a in acc.iter_mut() {
                        *a += 255.0 * wgt;
                    }
                } else {
                    let o = (py as usize * self.w + px as usize) * 3;
                    for (a, &v) in acc.iter_mut().zip(&self.data[o..o + 3]) {
                        *a += v as f32 * wgt;
                    }
                }
            }
        }
        *out = acc;
    }
}

/// Axis-aligned pixel rectangle `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixRect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl PixRect {
    pub fn full(w: usize, h: usize) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }
    }

    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }

    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }

    /// Grown by `m` pixels on every side, clipped to a `w × h` image.
    pub fn expand(&self, m: usize, w: usize, h: usize) -> Self {
        Self {
            x0: self.x0.saturating_sub(m),
            y0: self.y0.saturating_sub(m),
            x1: (self.x1 + m).min(w),
            y1: (self.y1 + m).min(h),
        }
    }
}

/// Luminance difference from the page background that counts as ink.
pub const INK_DELTA: i32 = 40;

/// Bounding box of the page content: connected (8-neighbour) blobs of
/// pixels whose luminance differs from the background (the most common
/// luminance) by more than [`INK_DELTA`], ignoring blobs smaller than
/// [`min_ink_blob`]. Specks of scanner grain or JPEG noise are scattered
/// over the whole page but are only a pixel or two each, while glyphs are
/// far larger; counting them would stretch the box to the full page.
pub fn ink_bounds(img: &Rgb<'_>) -> InkScan {
    const UNSEEN: u8 = 1;
    const SEEN: u8 = 2;
    let (w, h) = (img.w, img.h);
    let luma = |p: &[u8]| (77 * p[0] as i32 + 150 * p[1] as i32 + 29 * p[2] as i32) >> 8;
    let px = img.data.chunks_exact(3).take(w * h);
    let mut hist = [0u32; 256];
    for p in px.clone() {
        hist[luma(p) as usize] += 1;
    }
    let Some(bg) = (0..256).max_by_key(|&i| hist[i]) else {
        return InkScan::default();
    };
    let bg = bg as i32;
    // 0 = background, UNSEEN = ink not yet in a blob, SEEN = ink in one.
    let mut state: Vec<u8> = px
        .map(|p| {
            if (luma(p) - bg).abs() > INK_DELTA {
                UNSEEN
            } else {
                0
            }
        })
        .collect();
    let min_blob = min_ink_blob(w, h);
    let mut r: Option<PixRect> = None;
    let mut specks = 0;
    let mut stack = Vec::new();
    for start in 0..state.len() {
        if state[start] != UNSEEN {
            continue;
        }
        state[start] = SEEN;
        stack.push(start);
        let (x, y) = (start % w, start / w);
        let mut n = 0usize;
        let mut blob = PixRect {
            x0: x,
            y0: y,
            x1: x + 1,
            y1: y + 1,
        };
        while let Some(i) = stack.pop() {
            n += 1;
            let (x, y) = (i % w, i / w);
            blob.x0 = blob.x0.min(x);
            blob.x1 = blob.x1.max(x + 1);
            blob.y0 = blob.y0.min(y);
            blob.y1 = blob.y1.max(y + 1);
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = ny * w + nx;
                    if state[j] == UNSEEN {
                        state[j] = SEEN;
                        stack.push(j);
                    }
                }
            }
        }
        if n < min_blob {
            specks += 1;
            continue;
        }
        r = Some(match r {
            None => blob,
            Some(b) => PixRect {
                x0: b.x0.min(blob.x0),
                y0: b.y0.min(blob.y0),
                x1: b.x1.max(blob.x1),
                y1: b.y1.max(blob.y1),
            },
        });
    }
    InkScan { bounds: r, specks }
}

/// Specks (ink blobs too small to be content) per million pixels above
/// which a page counts as grainy. Clean scans and renders have a handful
/// (stray antialiasing fragments); film grain or JPEG noise, hundreds.
pub const GRAIN_SPECKS_PER_MPX: usize = 64;

/// What [`ink_bounds`] found on a page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InkScan {
    /// Bounding box of the content; `None` when nothing qualifies.
    pub bounds: Option<PixRect>,
    /// Ink blobs ignored as too small to be content.
    pub specks: usize,
}

impl InkScan {
    /// Whether a `w × h` page is covered in noise specks (see
    /// [`GRAIN_SPECKS_PER_MPX`]).
    pub fn grainy(&self, w: usize, h: usize) -> bool {
        self.specks * 1_000_000 > GRAIN_SPECKS_PER_MPX * w * h
    }
}

/// Smallest ink blob (in pixels) that counts as content on a `w × h`
/// page: grows with the square of the rendering scale, since a speck of
/// grain covers more pixels on a finer raster, but never below 3 pixels.
/// About 10 pixels on a letter page at 150 dpi, where a full stop is ~9
/// and a capital letter well over 50.
pub fn min_ink_blob(w: usize, h: usize) -> usize {
    let s = w.max(h) / 512;
    (s * s).max(3)
}

/// Sample the rectangle `r` (reading axis horizontal) into an
/// `out_w × out_h` RGB f32 buffer with values in 0..=255, planar CHW.
/// `flip` rotates the crop by 180° (for upside-down text).
pub fn crop_rotrect(
    img: &Rgb<'_>,
    r: &RotRect,
    out_w: usize,
    out_h: usize,
    flip: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; 3 * out_w * out_h];
    let plane = out_w * out_h;
    let mut px = [0f32; 3];
    for j in 0..out_h {
        let mut t = ((j as f64 + 0.5) / out_h as f64 - 0.5) * r.h;
        if flip {
            t = -t;
        }
        for i in 0..out_w {
            let mut s = ((i as f64 + 0.5) / out_w as f64 - 0.5) * r.w;
            if flip {
                s = -s;
            }
            let (x, y) = r.at(s, t);
            img.sample(x, y, &mut px);
            let o = j * out_w + i;
            out[o] = px[0];
            out[plane + o] = px[1];
            out[2 * plane + o] = px[2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    /// `w × h` RGB image of colour `bg` with `ink` pixels set to `fg`.
    fn page(w: usize, h: usize, bg: u8, fg: u8, ink: &[(usize, usize)]) -> Vec<u8> {
        let mut d = vec![bg; w * h * 3];
        for &(x, y) in ink {
            d[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(fg);
        }
        d
    }

    #[test]
    fn ink_bounds_finds_content_and_ignores_specks() {
        // Two short strokes of text, plus a lone speck near the corner.
        let mut ink: Vec<(usize, usize)> = (20..40).flat_map(|x| [(x, 30), (x, 31)]).collect();
        ink.extend((25..60).map(|x| (x, 70)));
        ink.push((95, 5));
        let d = page(100, 100, 250, 20, &ink);
        let scan = ink_bounds(&Rgb {
            data: &d,
            w: 100,
            h: 100,
        });
        assert_eq!(scan.specks, 1);
        assert!(
            scan.grainy(100, 100),
            "1 speck in 10k pixels is 100 per Mpx"
        );
        let r = scan.bounds.unwrap();
        assert_eq!(
            r,
            PixRect {
                x0: 20,
                y0: 30,
                x1: 60,
                y1: 71
            }
        );
        // Light text on a dark page works the same way.
        let d = page(100, 100, 10, 230, &ink);
        let dark = Rgb {
            data: &d,
            w: 100,
            h: 100,
        };
        assert_eq!(ink_bounds(&dark).bounds, Some(r));
        // Faint texture below the ink threshold is background.
        let d = page(100, 100, 250, 230, &ink);
        let faint = Rgb {
            data: &d,
            w: 100,
            h: 100,
        };
        assert_eq!(ink_bounds(&faint), InkScan::default());
    }

    #[test]
    fn ink_bounds_ignores_grain_spread_over_the_page() {
        // A paragraph-sized block of text in the top-left quarter...
        let mut ink: Vec<(usize, usize)> = (30..90)
            .flat_map(|x| (40..44).map(move |y| (x, y)))
            .collect();
        // ... a thin diagonal stroke (8-connected only) below it ...
        ink.extend((0..12).map(|i| (30 + i, 60 + i)));
        // ... and two-pixel grain specks in every corner of the page.
        for &(x, y) in &[(2, 2), (195, 3), (4, 196), (190, 190), (150, 100)] {
            ink.extend([(x, y), (x + 1, y)]);
        }
        ink.extend([(120, 150), (121, 151)]); // diagonal pair
        let d = page(200, 200, 245, 30, &ink);
        let r = ink_bounds(&Rgb {
            data: &d,
            w: 200,
            h: 200,
        })
        .bounds
        .unwrap();
        assert_eq!(
            r,
            PixRect {
                x0: 30,
                y0: 40,
                x1: 90,
                y1: 72
            }
        );
        // Grain alone is no content.
        let grain: Vec<(usize, usize)> = (0..20)
            .flat_map(|i| [(i * 9, i * 7), (i * 9 + 1, i * 7)])
            .collect();
        let d = page(200, 200, 245, 30, &grain);
        let noisy = Rgb {
            data: &d,
            w: 200,
            h: 200,
        };
        let scan = ink_bounds(&noisy);
        assert_eq!(scan.bounds, None);
        assert_eq!(scan.specks, 20);
        assert!(scan.grainy(200, 200));
        assert!(!scan.grainy(2000, 2000), "5 specks per Mpx is clean");
    }

    #[test]
    fn min_ink_blob_scales_with_the_raster() {
        assert_eq!(min_ink_blob(100, 100), 3);
        assert_eq!(min_ink_blob(1275, 1650), 9);
        assert_eq!(min_ink_blob(2550, 3300), 36);
    }

    #[test]
    fn pixrect_expand_clips_to_image() {
        let r = PixRect {
            x0: 5,
            y0: 40,
            x1: 60,
            y1: 90,
        };
        assert_eq!(
            r.expand(10, 100, 95),
            PixRect {
                x0: 0,
                y0: 30,
                x1: 70,
                y1: 95
            }
        );
        assert_eq!(PixRect::full(7, 9).width(), 7);
        assert_eq!(PixRect::full(7, 9).height(), 9);
    }

    #[test]
    fn hull_of_square_with_interior_point() {
        let mut pts = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)];
        let h = convex_hull(&mut pts);
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn min_rect_of_axis_aligned_box() {
        let mut pts = vec![(10.0, 5.0), (50.0, 5.0), (50.0, 15.0), (10.0, 15.0)];
        let r = min_area_rect(&convex_hull(&mut pts)).unwrap();
        assert!(approx(r.w, 40.0, 1e-9) && approx(r.h, 10.0, 1e-9));
        assert!(approx(r.cx, 30.0, 1e-9) && approx(r.cy, 10.0, 1e-9));
        assert!(approx(r.angle, 0.0, 1e-9));
        assert!(!r.is_vertical());
    }

    #[test]
    fn min_rect_recovers_rotation() {
        let truth = RotRect {
            cx: 100.0,
            cy: 80.0,
            w: 120.0,
            h: 20.0,
            angle: 0.1,
        };
        let mut pts = truth.corners().to_vec();
        let r = min_area_rect(&convex_hull(&mut pts)).unwrap();
        assert!(approx(r.w, 120.0, 1e-6) && approx(r.h, 20.0, 1e-6));
        assert!(approx(r.angle, 0.1, 1e-6), "angle {}", r.angle);
        assert!(approx(r.cx, 100.0, 1e-6) && approx(r.cy, 80.0, 1e-6));
    }

    #[test]
    fn vertical_rect_is_flagged() {
        let mut pts = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (0.0, 100.0)];
        let r = min_area_rect(&convex_hull(&mut pts)).unwrap();
        assert!(r.is_vertical());
        assert!(approx(r.w, 100.0, 1e-9));
    }

    #[test]
    fn unclip_grows_every_side() {
        let r = RotRect {
            cx: 0.0,
            cy: 0.0,
            w: 40.0,
            h: 10.0,
            angle: 0.0,
        };
        let g = unclip(r, 1.5);
        // d = 400 * 1.5 / 100 = 6
        assert!(approx(g.w, 52.0, 1e-9) && approx(g.h, 22.0, 1e-9));
    }

    #[test]
    fn db_finds_two_separate_lines() {
        let (w, h) = (60usize, 30usize);
        let mut prob = vec![0f32; w * h];
        for y in 5..9 {
            for x in 5..50 {
                prob[y * w + x] = 0.9;
            }
        }
        for y in 18..22 {
            for x in 10..40 {
                prob[y * w + x] = 0.8;
            }
        }
        let mut boxes = db_boxes(&prob, w, h, &DbParams::default());
        boxes.sort_by(|a, b| a.rect.cy.partial_cmp(&b.rect.cy).unwrap());
        assert_eq!(boxes.len(), 2);
        assert!(approx(boxes[0].rect.cy, 7.0, 0.01));
        assert!(boxes[0].rect.w > 45.0 && boxes[0].rect.h > 4.0);
        assert!(approx(boxes[1].score, 0.8, 1e-6));
    }

    #[test]
    fn db_drops_low_confidence_regions() {
        let (w, h) = (20usize, 20usize);
        let mut prob = vec![0f32; w * h];
        for y in 5..10 {
            for x in 2..18 {
                prob[y * w + x] = 0.4; // above thresh, below box_thresh
            }
        }
        assert!(db_boxes(&prob, w, h, &DbParams::default()).is_empty());
    }

    #[test]
    fn crop_reads_pixels_in_reading_order() {
        // 4×2 image: left half black, right half white.
        let mut data = vec![255u8; 4 * 2 * 3];
        for y in 0..2 {
            for x in 0..2 {
                let o = (y * 4 + x) * 3;
                data[o..o + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
        let img = Rgb {
            data: &data,
            w: 4,
            h: 2,
        };
        let r = RotRect {
            cx: 2.0,
            cy: 1.0,
            w: 4.0,
            h: 2.0,
            angle: 0.0,
        };
        let c = crop_rotrect(&img, &r, 4, 2, false);
        assert_eq!(c[0], 0.0);
        assert_eq!(c[3], 255.0);
        let f = crop_rotrect(&img, &r, 4, 2, true);
        assert_eq!(f[0], 255.0);
        assert_eq!(f[3], 0.0);
    }
}
