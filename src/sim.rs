use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::car::{Car, CarParams};
use crate::map::OccGrid;

const N_LOOK: usize = 10;

pub struct Obs<'a> {
    pub scans: &'a [f64],
    pub rewards: &'a [f64],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
    pub state: &'a [f64],
}

pub struct Sim {
    pub map: OccGrid,
    pub cars: Vec<Car>,
    pub dr_frac: f64,
    pub dt: f64,
    pub ds: usize,
    pub n_beams: usize,
    pub fov: f64,
    pub max_range: f64,
    pub min_range: f64,
    pub obs_dim: usize,
    pub rngs: Vec<SmallRng>,
    pub waypoint_idx: Vec<usize>,
    pub steps: Vec<u32>,
    pub max_steps: u32,
    look_step: usize,
    beam_sin_cos: Vec<(f64, f64)>,
    buf_terminated: Vec<bool>,
    buf_truncated: Vec<bool>,
    buf_rewards: Vec<f64>,
    buf_scans: Vec<f64>,
    buf_state: Vec<f64>,
}

impl Sim {
    pub fn new(yaml: &str, n: usize, max_steps: u32) -> Self {
        let n_beams: usize = 108;
        let fov: f64 = 270.0 * PI / 180.0;
        let beam_sin_cos: Vec<(f64, f64)> = (0..n_beams)
            .map(|i| -fov / 2.0 + fov * i as f64 / (n_beams - 1) as f64)
            .map(|a| a.sin_cos())
            .collect();
        let map = OccGrid::load(yaml);
        let n_pts = map.skeleton.points.len();
        let avg_sp: f64 = (0..n_pts)
            .map(|i| {
                let [x1, y1] = map.skeleton.points[i];
                let [x2, y2] = map.skeleton.points[(i + 1) % n_pts];
                ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
            })
            .sum::<f64>()
            / n_pts as f64;
        let look_step = (1.0 / avg_sp).round().max(1.0) as usize;
        let obs_dim = n_beams + 3 + 2 + N_LOOK * 2;
        let (px, py, th) = map.skeleton.get_point(0);
        Self {
            map,
            cars: vec![Car::new(px, py, th, 0.0, CarParams::default()); n],
            dr_frac: 0.2,
            dt: 1.0 / 60.0,
            ds: 6,
            n_beams,
            fov,
            max_range: 10.0,
            min_range: 0.06,
            obs_dim,
            rngs: (0..n).map(|i| SmallRng::seed_from_u64(i as u64)).collect(),
            waypoint_idx: vec![0; n],
            steps: vec![0; n],
            max_steps,
            look_step,
            beam_sin_cos,
            buf_terminated: vec![false; n],
            buf_truncated: vec![false; n],
            buf_rewards: vec![0.0; n],
            buf_scans: vec![0.0; n * obs_dim],
            buf_state: vec![0.0; n * 7],
        }
    }

    pub fn seed(&mut self, seed: u64) {
        self.rngs = (0..self.rngs.len())
            .map(|i| SmallRng::seed_from_u64(seed + i as u64))
            .collect();
    }

    pub fn reset(&mut self) -> Obs<'_> {
        self.steps.fill(0);
        for (i, c) in self.cars.iter_mut().enumerate() {
            let ri = self.rngs[i].random_range(0..self.map.skeleton.points.len());
            let (px, py, th) = self.map.skeleton.get_point(ri);
            *c = Car::new(
                px,
                py,
                th,
                0.0,
                CarParams::random(&mut self.rngs[i], self.dr_frac),
            );
            self.waypoint_idx[i] = ri;
        }
        self.observe()
    }

    fn tick(&mut self, actions: Option<&[f64]>) -> Obs<'_> {
        let (n_wps, nb, od) = (self.map.skeleton.points.len(), self.n_beams, self.obs_dim);
        let (mr, mn, ms) = (self.max_range, self.min_range, self.max_steps);
        let (dt, ds, df, ls) = (self.dt, self.ds, self.dr_frac, self.look_step);
        let (map, bsc) = (&self.map, &self.beam_sin_cos);

        self.buf_terminated
            .par_iter_mut()
            .zip(self.buf_truncated.par_iter_mut())
            .zip(self.buf_rewards.par_iter_mut())
            .zip(self.cars.par_iter_mut())
            .zip(self.waypoint_idx.par_iter_mut())
            .zip(self.steps.par_iter_mut())
            .zip(self.buf_scans.par_chunks_mut(od))
            .zip(self.buf_state.par_chunks_mut(7))
            .zip(self.rngs.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, ((((((((term, trunc), rew), car), wi), step), scan), st), rng))| {
                    if let Some(a) = actions {
                        *step += 1;
                        for _ in 0..ds {
                            car.step(a[i * 2], a[i * 2 + 1], dt / ds as f64);
                        }
                    }
                    *term = map.car_collides(car);
                    *trunc = *step >= ms;

                    let prev = *wi;
                    let (px, py) = map.position_to_pixels(car.x, car.y);
                    *wi = map.skeleton.get_idx(px, py);
                    let mut d = *wi as f64 - prev as f64;
                    if d > n_wps as f64 / 2.0 {
                        d -= n_wps as f64;
                    } else if d < -(n_wps as f64 / 2.0) {
                        d += n_wps as f64;
                    }
                    *rew = d / n_wps as f64 * 100.0 * (1.0 + car.velocity.max(0.0) / 10.0)
                        - 0.1 * (-3.0 * map.edt(px, py)).exp()
                        - if *term { 100.0 } else { 0.0 };

                    if *term || *trunc {
                        let ri = rng.random_range(0..n_wps);
                        let (px, py, th) = map.skeleton.get_point(ri);
                        *step = 0;
                        *car = Car::new(px, py, th, 0.0, CarParams::random(rng, df));
                        *wi = ri;
                    }

                    let (sh, ch) = car.theta.sin_cos();
                    let (lx, ly) = (car.x + car.params.lf * ch, car.y + car.params.lf * sh);
                    for (j, &(sa, ca)) in bsc.iter().enumerate() {
                        scan[j] = map
                            .raycast(lx, ly, ch * ca - sh * sa, sh * ca + ch * sa, mn, mr)
                            .clamp(mn, mr);
                    }
                    scan[nb] = car.velocity;
                    scan[nb + 1] = car.steering;
                    scan[nb + 2] = car.yaw_rate;

                    let (cx, cy, cth) = map.skeleton.get_point(*wi);
                    let b = nb + 3;
                    scan[b] = ((cth - car.theta) + PI).rem_euclid(2.0 * PI) - PI;
                    scan[b + 1] = -(car.x - cx) * cth.sin() + (car.y - cy) * cth.cos();
                    for k in 0..N_LOOK {
                        let [wx, wy] = map.skeleton.points[(*wi + (k + 1) * ls) % n_wps];
                        let (dx, dy) = (wx - car.x, wy - car.y);
                        scan[b + 2 + k * 2] = dx * ch + dy * sh;
                        scan[b + 3 + k * 2] = -dx * sh + dy * ch;
                    }

                    st[0] = car.x;
                    st[1] = car.y;
                    st[2] = car.theta;
                    st[3] = car.velocity;
                    st[4] = car.steering;
                    st[5] = car.yaw_rate;
                    st[6] = car.slip_angle;
                },
            );

        Obs {
            scans: &self.buf_scans,
            rewards: &self.buf_rewards,
            terminated: &self.buf_terminated,
            truncated: &self.buf_truncated,
            state: &self.buf_state,
        }
    }

    pub fn step(&mut self, actions: &[f64]) -> Obs<'_> {
        self.tick(Some(actions))
    }
    pub fn observe(&mut self) -> Obs<'_> {
        self.tick(None)
    }
}
