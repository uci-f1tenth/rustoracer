use crate::car::{Car, LENGTH, WIDTH};
use crate::skeleton::{extract_main_loop, thin_image_edges};
use image::GrayImage;
use imageproc::distance_transform::euclidean_squared_distance_transform;
use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use serde::Deserialize;
#[cfg(feature = "show_images")]
use show_image::{ImageInfo, ImageView, create_window};

#[derive(Deserialize)]
struct MapMeta {
    image: String,
    resolution: f64,
    origin: [f64; 3],
}

pub struct OccGrid {
    inv_res: f64,
    pub img: GrayImage,
    pub edt: Vec<f64>,
    pub ordered_skeleton: Vec<[f64; 2]>,
    pub skeleton_lut: Vec<usize>,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

#[cfg(feature = "show_images")]
fn view_image(img: &GrayImage, title: &str) {
    let window = create_window(title, Default::default()).unwrap();
    let image_view = ImageView::new(ImageInfo::mono8(img.width(), img.height()), img.as_raw());
    window.set_image(title, image_view).unwrap();
}

impl OccGrid {
    pub fn load(yaml: &str) -> Self {
        let m: MapMeta = serde_saphyr::from_str(&std::fs::read_to_string(yaml).unwrap()).unwrap();
        let dir = std::path::Path::new(yaml).parent().unwrap();
        let img = image::open(dir.join(&m.image)).unwrap().into_luma8();
        let mut occupied_image = img.clone();
        for pixel in occupied_image.pixels_mut() {
            pixel.0[0] = if pixel.0[0] < 128 { 255 } else { 0 };
        }
        let edt = euclidean_squared_distance_transform(&occupied_image);
        let mut skeleton = thin_image_edges(&occupied_image);
        let ordered_skeleton =
            extract_main_loop(&mut skeleton, m.resolution, m.origin[0], m.origin[1]);
        #[cfg(feature = "show_images")]
        view_image(&occupied_image, "occupied");
        #[cfg(feature = "show_images")]
        view_image(&skeleton, "skeleton");
        let skeleton_tree = ImmutableKdTree::new_from_slice(&ordered_skeleton);
        let (w, h) = (img.width(), img.height());
        let skeleton_lut =
            Self::build_skeleton_lut(&skeleton_tree, w, h, m.resolution, m.origin[0], m.origin[1]);
        Self {
            inv_res: 1.0 / m.resolution,
            img,
            edt: edt.pixels().map(|p| p.0[0].sqrt() * m.resolution).collect(),
            ordered_skeleton,
            skeleton_lut,
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        }
    }

    fn build_skeleton_lut(
        tree: &ImmutableKdTree<f64, usize, 2, 32>,
        w: u32,
        h: u32,
        res: f64,
        ox: f64,
        oy: f64,
    ) -> Vec<usize> {
        (0..h)
            .flat_map(|py| {
                (0..w).map(move |px| {
                    let x = px as f64 * res + ox;
                    let y = (h - 1 - py) as f64 * res + oy;
                    tree.nearest_one::<SquaredEuclidean>(&[x, y]).item
                })
            })
            .collect()
    }

    pub fn skeleton_idx(&self, x: f64, y: f64) -> usize {
        let (px, py) = self.position_to_pixels(x, y);
        self.skeleton_lut[(py * self.img.width() + px) as usize]
    }

    #[inline]
    pub fn position_to_pixels(&self, x: f64, y: f64) -> (u32, u32) {
        let px = ((x - self.ox) * self.inv_res) as u32;
        let py = self.img.height() - 1 - ((y - self.oy) * self.inv_res) as u32;
        (px, py)
    }

    #[inline]
    pub fn edt(&self, px: u32, py: u32) -> f64 {
        if (0..self.img.width()).contains(&px) && (0..self.img.height()).contains(&py) {
            unsafe {
                *self
                    .edt
                    .get_unchecked((px + py * self.img.width()) as usize)
            }
        } else {
            0.0
        }
    }

    #[inline]
    pub fn raycast(&self, x: f64, y: f64, dx: f64, dy: f64, max: f64) -> f64 {
        let px0 = (x - self.ox) * self.inv_res;
        let py0 = (self.img.height() - 1) as f64 - (y - self.oy) * self.inv_res;
        let dpx = dx * self.inv_res;
        let dpy = -dy * self.inv_res;

        let w = self.img.width();
        let h = self.img.height();

        let mut t = 0.0;
        while t < max {
            let pxi = (px0 + t * dpx) as u32;
            let pyi = (py0 + t * dpy) as u32;

            let d = if pxi < w && pyi < h {
                unsafe { *self.edt.get_unchecked((pxi + pyi * w) as usize) }
            } else {
                return t;
            };

            if d < self.res {
                return t;
            }
            t += d.max(self.res);
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
                if (fx * ca - fy * sa).abs() <= hl && (fx * sa + fy * ca).abs() <= hw {
                    Some(((cx as i32 + dx) as u32, (cy as i32 + dy) as u32))
                } else {
                    None
                }
            })
        })
    }

    pub fn car_collides(&self, car: &Car) -> bool {
        if !car.x.is_finite() || !car.y.is_finite() || !car.theta.is_finite() {
            return true;
        }
        let (px, py) = self.position_to_pixels(car.x, car.y);
        let center_clearance = self.edt(px, py);
        const CAR_CIRCUMRADIUS: f64 = 0.328824; // sqrt((LENGTH/2)^2 + (WIDTH/2)^2)

        if center_clearance > CAR_CIRCUMRADIUS {
            return false;
        }
        self.car_pixels(car)
            .into_iter()
            .any(|(x, y)| self.edt(x, y) < self.res)
    }
}
