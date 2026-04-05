use image::{GrayImage, Luma};
use std::collections::{HashMap, HashSet, VecDeque};

const BG: u8 = 255;
const DIRS: [(i32, i32); 8] = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

pub fn thin_image_edges(img_in: &GrayImage) -> GrayImage {
    let mut img = img_in.clone();
    let (w, h) = (img.width(), img.height());
    let mut to_remove = Vec::with_capacity(1024);
    let mut phase_one = true;

    loop {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if img.get_pixel(x, y).0[0] == BG {
                    continue;
                }

                let p = |dx: i32, dy: i32| {
                    img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)
                        .0[0]
                        != BG
                };
                let n = [
                    p(0, -1),
                    p(1, -1),
                    p(1, 0),
                    p(1, 1),
                    p(0, 1),
                    p(-1, 1),
                    p(-1, 0),
                    p(-1, -1),
                ];
                let filled = n.iter().filter(|&&v| v).count();
                if !(2..=7).contains(&filled) {
                    continue;
                }

                let trans = (0..8).filter(|&i| !n[i] && n[(i + 1) & 7]).count();
                if trans != 1 && trans != 2 {
                    continue;
                }

                let [p2, p3, p4, p5, p6, p7, p8, p9] = n;
                let remove = if phase_one {
                    if trans == 1 {
                        !p4 || !p6 || (!p2 && !p8)
                    } else {
                        (p2 && p4 && !p6 && !p7 && !p8) || (p4 && p6 && !p2 && !p8 && !p9)
                    }
                } else {
                    if trans == 1 {
                        !p2 || !p8 || (!p4 && !p6)
                    } else {
                        (p2 && p8 && !p4 && !p5 && !p6) || (p6 && p8 && !p2 && !p3 && !p4)
                    }
                };
                if remove {
                    to_remove.push((x, y));
                }
            }
        }

        phase_one = !phase_one;
        if to_remove.is_empty() {
            return img;
        }
        for &(x, y) in &to_remove {
            img.put_pixel(x, y, Luma([BG]));
        }
        to_remove.clear();
    }
}

fn adj(fg: &HashSet<(u32, u32)>, x: u32, y: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
    DIRS.iter().filter_map(move |&(dx, dy)| {
        let n = ((x as i32 + dx) as u32, (y as i32 + dy) as u32);
        fg.contains(&n).then_some(n)
    })
}

pub fn extract_main_loop(img: &mut GrayImage, res: f64, ox: f64, oy: f64) -> Vec<[f64; 2]> {
    let fg: HashSet<(u32, u32)> = img
        .enumerate_pixels()
        .filter(|(_, _, p)| p.0[0] != BG)
        .map(|(x, y, _)| (x, y))
        .collect();

    let origin_px = (-ox / res, (img.height() - 1) as f64 + oy / res);
    let start = *fg
        .iter()
        .min_by_key(|&&(x, y)| {
            let (dx, dy) = (x as f64 - origin_px.0, y as f64 - origin_px.1);
            (dx * dx + dy * dy) as i64
        })
        .unwrap();

    let nbrs: Vec<_> = adj(&fg, start.0, start.1).collect();
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

fn morph_op(img: &GrayImage, radius: u32, is_dilate: bool) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    let r = radius as i32;
    GrayImage::from_fn(w, h, |x, y| {
        let hit = (-r..=r).any(|dy| {
            (-r..=r).any(|dx| {
                if dx * dx + dy * dy > r * r {
                    return is_dilate;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    return is_dilate;
                }
                let fg = img.get_pixel(nx as u32, ny as u32).0[0] != BG;
                if is_dilate { fg } else { !fg }
            })
        });
        let fg = if is_dilate { hit } else { !hit };
        Luma([if fg { 0 } else { BG }])
    })
}

pub fn morphological_open(img: &GrayImage, radius: u32) -> GrayImage {
    morph_op(&morph_op(img, radius, false), radius, true)
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

    let mut ata = vec![vec![0.0; d + 1]; d];
    for i in 0..d {
        for j in 0..d {
            ata[i][j] = (0..window).map(|k| a[k][i] * a[k][j]).sum();
        }
    }
    ata[0][d] = 1.0;

    for col in 0..d {
        let max_row = (col..d)
            .max_by(|&i, &j| ata[i][col].abs().total_cmp(&ata[j][col].abs()))
            .unwrap();
        ata.swap(col, max_row);
        let pivot = ata[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..d {
            let f = ata[row][col] / pivot;
            for k in col..=d {
                ata[row][k] -= f * ata[col][k];
            }
        }
    }

    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        x[i] = ata[i][d];
        for j in (i + 1)..d {
            x[i] -= ata[i][j] * x[j];
        }
        x[i] /= ata[i][i];
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
