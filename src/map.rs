use std::f64::consts::PI;

use crate::car::{Car, LENGTH, WIDTH};
use crate::skeleton::{
    extract_main_loop, morphological_open, savitzky_golay_smooth, thin_image_edges,
};
use image::GrayImage;
use imageproc::distance_transform::euclidean_squared_distance_transform;
use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;

pub struct SkeletonConfig {
    pub open_radius: u32,
    pub sg_window: usize,
    pub sg_degree: usize,
}

impl Default for SkeletonConfig {
    fn default() -> Self {
        Self {
            open_radius: 2,
            sg_window: 11,
            sg_degree: 3,
        }
    }
}

#[derive(Deserialize)]
struct MapMeta {
    image: String,
    resolution: f64,
    origin: [f64; 3],
}

pub struct Skeleton {
    width: u32,
    pub points: Vec<[f64; 2]>,
    pub lut: Vec<usize>,
}

impl Skeleton {
    pub fn new(
        img: &GrayImage,
        res: f64,
        ox: f64,
        oy: f64,
        otheta: f64,
        cfg: &SkeletonConfig,
    ) -> Self {
        let mut thinned = thin_image_edges(img);
        let mut ordered = extract_main_loop(&mut thinned, res, ox, oy);
        assert!(ordered.len() >= 2);

        let diff = (ordered[1][1] - ordered[0][1]).atan2(ordered[1][0] - ordered[0][0]);
        if ((diff - otheta + 3.0 * PI) % (2.0 * PI) - PI).abs() > PI / 2.0 {
            ordered[1..].reverse();
        }

        let smoothed = savitzky_golay_smooth(&ordered, cfg.sg_window, cfg.sg_degree);
        let tree = ImmutableKdTree::new_from_slice(&smoothed);
        let lut = Self::build_lut(&tree, img.width(), img.height(), res, ox, oy);
        Self {
            points: smoothed,
            lut,
            width: img.width(),
        }
    }

    fn build_lut(
        tree: &ImmutableKdTree<f64, usize, 2, 32>,
        w: u32,
        h: u32,
        res: f64,
        ox: f64,
        oy: f64,
    ) -> Vec<usize> {
        (0..h)
            .into_par_iter()
            .flat_map(|py| {
                (0..w).into_par_iter().map(move |px| {
                    let x = px as f64 * res + ox;
                    let y = (h - 1 - py) as f64 * res + oy;
                    tree.nearest_one::<SquaredEuclidean>(&[x, y]).item
                })
            })
            .collect()
    }

    pub fn get_point(&self, idx: usize) -> (f64, f64, f64) {
        let [px, py] = self.points[idx];
        let [nx, ny] = self.points[(idx + 1) % self.points.len()];
        (px, py, (ny - py).atan2(nx - px))
    }

    #[inline]
    pub fn get_idx(&self, px: u32, py: u32) -> usize {
        self.lut[(py * self.width + px) as usize]
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}

pub struct OccGrid {
    inv_res: f64,
    pub img: GrayImage,
    pub edt: Vec<f64>,
    pub skeleton: Skeleton,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

impl OccGrid {
    pub fn load(yaml: &str) -> Self {
        Self::load_with_config(yaml, &SkeletonConfig::default())
    }

    pub fn load_with_config(yaml: &str, cfg: &SkeletonConfig) -> Self {
        let m: MapMeta = serde_saphyr::from_str(&std::fs::read_to_string(yaml).unwrap()).unwrap();
        let dir = std::path::Path::new(yaml).parent().unwrap();
        let img = image::open(dir.join(&m.image)).unwrap().into_luma8();

        let mut binary = img.clone();
        for p in binary.pixels_mut() {
            p.0[0] = if p.0[0] < 210 { 255 } else { 0 };
        }

        let edt_raw = euclidean_squared_distance_transform(&binary);
        let opened = morphological_open(&binary, cfg.open_radius);
        let skeleton = Skeleton::new(
            &opened,
            m.resolution,
            m.origin[0],
            m.origin[1],
            m.origin[2],
            cfg,
        );

        Self {
            inv_res: 1.0 / m.resolution,
            img,
            edt: edt_raw
                .pixels()
                .map(|p| p.0[0].sqrt() * m.resolution)
                .collect(),
            skeleton,
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        }
    }

    #[inline]
    pub fn position_to_pixels(&self, x: f64, y: f64) -> (u32, u32) {
        (
            ((x - self.ox) * self.inv_res) as u32,
            self.img.height() - 1 - ((y - self.oy) * self.inv_res) as u32,
        )
    }

    #[inline]
    pub fn edt(&self, px: u32, py: u32) -> f64 {
        if px < self.img.width() && py < self.img.height() {
            unsafe {
                *self
                    .edt
                    .get_unchecked((py * self.img.width() + px) as usize)
            }
        } else {
            0.0
        }
    }

    #[inline]
    pub fn raycast(&self, x: f64, y: f64, dx: f64, dy: f64, min: f64, max: f64) -> f64 {
        let px0 = (x - self.ox) * self.inv_res;
        let py0 = (self.img.height() - 1) as f64 - (y - self.oy) * self.inv_res;
        let (dpx, dpy) = (dx * self.inv_res, -dy * self.inv_res);
        let (w, h) = (self.img.width(), self.img.height());
        let mut t = min;
        while t < max {
            let (pxi, pyi) = ((px0 + t * dpx) as u32, (py0 + t * dpy) as u32);
            let d = if pxi < w && pyi < h {
                unsafe { *self.edt.get_unchecked((pxi + pyi * w) as usize) }
            } else {
                return t;
            };
            if d < self.res * 0.5 {
                return t;
            }
            t += d.max(self.res * 0.5);
        }
        max
    }

    pub fn car_pixels<'a>(&'a self, car: &'a Car) -> impl Iterator<Item = (u32, u32)> + 'a {
        let (sa, ca) = car.theta.sin_cos();
        let (hl, hw) = (LENGTH / 2.0 * self.inv_res, WIDTH / 2.0 * self.inv_res);
        let (cx, cy) = self.position_to_pixels(car.x, car.y);
        let r = hl.hypot(hw).ceil() as i32;
        (-r..=r).flat_map(move |dy| {
            (-r..=r).filter_map(move |dx| {
                let (fx, fy) = (dx as f64, dy as f64);
                ((fx * ca - fy * sa).abs() <= hl && (fx * sa + fy * ca).abs() <= hw)
                    .then_some(((cx as i32 + dx) as u32, (cy as i32 + dy) as u32))
            })
        })
    }

    pub fn car_collides(&self, car: &Car) -> bool {
        if !car.x.is_finite() || !car.y.is_finite() || !car.theta.is_finite() {
            return true;
        }
        let (px, py) = self.position_to_pixels(car.x, car.y);
        const R_SQ: f64 = (LENGTH / 2.0) * (LENGTH / 2.0) + (WIDTH / 2.0) * (WIDTH / 2.0);
        let c = self.edt(px, py);
        if c * c > R_SQ {
            return false;
        }
        self.car_pixels(car)
            .any(|(px, py)| self.edt(px, py) < self.res)
    }
}
