use image::{GrayImage, Luma};
use std::collections::{HashMap, HashSet, VecDeque};

const BG: u8 = 255;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Edge {
    Empty = 0,
    Filled = 1,
    DoesNotExist,
}

pub struct NeighborInfo {
    pub filled: u8,
    pub neighbors: u8,
    pub edge_status: [Edge; 8],
}

pub fn get_neighbor_info(img: &GrayImage, width: u32, height: u32, x: u32, y: u32) -> NeighborInfo {
    let mut filled = 0u8;
    let mut neighbors = 0u8;

    macro_rules! check {
        ($cond:expr, $px:expr, $py:expr) => {
            if $cond {
                neighbors += 1;
                if img.get_pixel($px, $py)[0] != BG {
                    filled += 1;
                    Edge::Filled
                } else {
                    Edge::Empty
                }
            } else {
                Edge::DoesNotExist
            }
        };
    }

    let p9 = check!(y > 0 && x > 0, x - 1, y - 1);
    let p2 = check!(y > 0, x, y - 1);
    let p3 = check!(y > 0 && x + 1 < width, x + 1, y - 1);
    let p8 = check!(x > 0, x - 1, y);
    let p4 = check!(x + 1 < width, x + 1, y);
    let p7 = check!(x > 0 && y + 1 < height, x - 1, y + 1);
    let p6 = check!(y + 1 < height, x, y + 1);
    let p5 = check!(x + 1 < width && y + 1 < height, x + 1, y + 1);

    NeighborInfo {
        filled,
        neighbors,
        edge_status: [p2, p3, p4, p5, p6, p7, p8, p9],
    }
}

pub fn thin_image_edges(img_in: &GrayImage) -> GrayImage {
    let mut img = img_in.clone();
    let mut to_remove = Vec::new();
    let mut phase_one = true;

    loop {
        for (x, y, p) in img.enumerate_pixels() {
            if p.0[0] == BG {
                continue;
            }
            let info = get_neighbor_info(&img, img.width(), img.height(), x, y);
            let [p2, p3, p4, p5, p6, p7, p8, p9] = info.edge_status;
            if !(2..=7).contains(&info.filled) || info.neighbors != 8 {
                continue;
            }

            let transitions = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                .windows(2)
                .filter(|w| w[0] == Edge::Empty && w[1] == Edge::Filled)
                .count();

            if !(transitions == 1 || transitions == 2) {
                continue;
            }

            let remove = if phase_one {
                if transitions == 1 {
                    p4 == Edge::Empty
                        || p6 == Edge::Empty
                        || (p2 == Edge::Empty && p8 == Edge::Empty)
                } else {
                    (p2 == Edge::Filled
                        && p4 == Edge::Filled
                        && p6 == Edge::Empty
                        && p7 == Edge::Empty
                        && p8 == Edge::Empty)
                        || (p4 == Edge::Filled
                            && p6 == Edge::Filled
                            && p2 == Edge::Empty
                            && p8 == Edge::Empty
                            && p9 == Edge::Empty)
                }
            } else {
                if transitions == 1 {
                    p2 == Edge::Empty
                        || p8 == Edge::Empty
                        || (p4 == Edge::Empty && p6 == Edge::Empty)
                } else {
                    (p2 == Edge::Filled
                        && p8 == Edge::Filled
                        && p4 == Edge::Empty
                        && p5 == Edge::Empty
                        && p6 == Edge::Empty)
                        || (p6 == Edge::Filled
                            && p8 == Edge::Filled
                            && p2 == Edge::Empty
                            && p3 == Edge::Empty
                            && p4 == Edge::Empty)
                }
            };

            if remove {
                to_remove.push((x, y));
            }
        }

        phase_one = !phase_one;
        for &(x, y) in &to_remove {
            img.put_pixel(x, y, Luma([BG]));
        }
        if to_remove.is_empty() {
            return img;
        }
        to_remove.clear();
    }
}

fn adj(fg: &HashSet<(u32, u32)>, x: u32, y: u32) -> Vec<(u32, u32)> {
    [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, 1),
        (0, -1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]
    .iter()
    .filter_map(|&(dx, dy)| {
        let n = ((x as i32 + dx) as u32, (y as i32 + dy) as u32);
        fg.contains(&n).then_some(n)
    })
    .collect()
}

pub fn extract_main_loop(img: &mut GrayImage, res: f64, ox: f64, oy: f64) -> Vec<[f64; 2]> {
    let fg: HashSet<(u32, u32)> = img
        .enumerate_pixels()
        .filter(|(_, _, p)| p.0[0] != BG)
        .map(|(x, y, _)| (x, y))
        .collect();

    let origin_px = (-ox / res, (img.height() - 1) as f64 + oy / res);
    let start = fg
        .iter()
        .copied()
        .min_by_key(|&(x, y)| {
            let (dx, dy) = (x as f64 - origin_px.0, y as f64 - origin_px.1);
            (dx * dx + dy * dy) as i64
        })
        .unwrap();

    let nbrs = adj(&fg, start.0, start.1);
    if nbrs.len() < 2 {
        return Vec::new();
    }

    let src = nbrs[0];
    let targets: HashSet<_> = nbrs[1..].iter().copied().collect();
    let mut parent: HashMap<(u32, u32), (u32, u32)> = HashMap::from([(src, src)]);
    let mut q = VecDeque::from([src]);
    let mut found = None;

    'bfs: while let Some(cur) = q.pop_front() {
        for n in adj(&fg, cur.0, cur.1) {
            if n == start || parent.contains_key(&n) {
                continue;
            }
            parent.insert(n, cur);
            if targets.contains(&n) {
                found = Some(n);
                break 'bfs;
            }
            q.push_back(n);
        }
    }

    let mut pixels = vec![start];
    if let Some(end) = found {
        let mut p = end;
        while p != src {
            pixels.push(p);
            p = parent[&p];
        }
        pixels.push(src);
    }
    pixels.reverse();

    let keep: HashSet<_> = pixels.iter().copied().collect();
    for (x, y, p) in img.enumerate_pixels_mut() {
        if p.0[0] != BG && !keep.contains(&(x, y)) {
            *p = Luma([BG]);
        }
    }

    let h = img.height();
    pixels
        .iter()
        .map(|&(px, py)| [px as f64 * res + ox, (h - 1 - py) as f64 * res + oy])
        .collect()
}

pub fn erode(img: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    let r = radius as i32;
    GrayImage::from_fn(w, h, |x, y| {
        let fg = (-r..=r).all(|dy| {
            (-r..=r).all(|dx| {
                if dx * dx + dy * dy > r * r {
                    return true;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    return false;
                }
                img.get_pixel(nx as u32, ny as u32).0[0] != BG
            })
        });
        Luma([if fg { 0 } else { BG }])
    })
}

pub fn dilate(img: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    let r = radius as i32;
    GrayImage::from_fn(w, h, |x, y| {
        let fg = (-r..=r).any(|dy| {
            (-r..=r).any(|dx| {
                if dx * dx + dy * dy > r * r {
                    return false;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    return false;
                }
                img.get_pixel(nx as u32, ny as u32).0[0] != BG
            })
        });
        Luma([if fg { 0 } else { BG }])
    })
}

pub fn morphological_open(img: &GrayImage, radius: u32) -> GrayImage {
    dilate(&erode(img, radius), radius)
}

fn sg_weights(window: usize, degree: usize) -> Vec<f64> {
    let half = window / 2;
    let d = degree + 1;

    let a: Vec<Vec<f64>> = (0..window)
        .map(|i| {
            let t = i as f64 - half as f64;
            (0..d).map(|j| t.powi(j as i32)).collect()
        })
        .collect();

    let mut ata = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..window {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    let mut aug: Vec<Vec<f64>> = ata
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(if i == 0 { 1.0 } else { 0.0 });
            r
        })
        .collect();

    for col in 0..d {
        let max_row = (col..d)
            .max_by(|&i, &j| aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap())
            .unwrap();
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..d {
            let f = aug[row][col] / pivot;
            for k in col..=d {
                aug[row][k] -= f * aug[col][k];
            }
        }
    }

    let mut x = vec![0.0f64; d];
    for i in (0..d).rev() {
        x[i] = aug[i][d];
        for j in (i + 1)..d {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    (0..window)
        .map(|i| (0..d).map(|j| a[i][j] * x[j]).sum())
        .collect()
}

pub fn savitzky_golay_smooth(points: &[[f64; 2]], window: usize, degree: usize) -> Vec<[f64; 2]> {
    let n = points.len();
    let h = sg_weights(window, degree);
    let half = window / 2;
    (0..n)
        .map(|i| {
            let (mut sx, mut sy) = (0.0, 0.0);
            for k in 0..window {
                let idx = (i + k + n - half) % n;
                sx += h[k] * points[idx][0];
                sy += h[k] * points[idx][1];
            }
            [sx, sy]
        })
        .collect()
}
