use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::car::{Car, CarParams};
use crate::map::OccGrid;

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
    pub rngs: Vec<SmallRng>,
    pub waypoint_idx: Vec<usize>,
    pub steps: Vec<u32>,
    pub max_steps: u32,
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
        let (px, py, th) = map.skeleton.get_point(0);
        Self {
            map,
            cars: vec![
                Car {
                    x: px,
                    y: py,
                    theta: th,
                    velocity: 0.0,
                    steering: 0.0,
                    yaw_rate: 0.0,
                    slip_angle: 0.0,
                    omega_f: 0.0,
                    omega_r: 0.0,
                    params: CarParams::default(),
                };
                n
            ],
            dr_frac: 0.2,
            dt: 1.0 / 60.0,
            ds: 6,
            n_beams,
            fov,
            max_range: 10.0,
            min_range: 0.06,
            rngs: (0..n).map(|i| SmallRng::seed_from_u64(i as u64)).collect(),
            waypoint_idx: vec![0; n],
            steps: vec![0; n],
            max_steps,
            beam_sin_cos,
            buf_terminated: vec![false; n],
            buf_truncated: vec![false; n],
            buf_rewards: vec![0.0; n],
            buf_scans: vec![0.0; n * (n_beams + 3)],
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
            *c = Car {
                x: px,
                y: py,
                theta: th,
                velocity: 0.0,
                steering: 0.0,
                yaw_rate: 0.0,
                slip_angle: 0.0,
                omega_f: 0.0,
                omega_r: 0.0,
                params: CarParams::random(&mut self.rngs[i], self.dr_frac),
            };
        }
        let (px, py) = self.map.position_to_pixels(0.0, 0.0);
        let nearest = self.map.skeleton.get_idx(px, py);
        self.waypoint_idx.fill(nearest);
        self.observe()
    }

    fn tick(&mut self, actions: Option<&[f64]>) -> Obs<'_> {
        let n_wps = self.map.skeleton.points.len();
        let n_beams = self.n_beams;
        let max_range = self.max_range;
        let min_range = self.min_range;
        let max_steps = self.max_steps;
        let dt = self.dt;
        let ds = self.ds;
        let map = &self.map;
        let beam_sin_cos = &self.beam_sin_cos;
        let dr_fract = self.dr_frac;

        self.buf_terminated
            .par_iter_mut()
            .zip(self.buf_truncated.par_iter_mut())
            .zip(self.buf_rewards.par_iter_mut())
            .zip(self.cars.par_iter_mut())
            .zip(self.waypoint_idx.par_iter_mut())
            .zip(self.steps.par_iter_mut())
            .zip(self.buf_scans.par_chunks_mut(n_beams + 3))
            .zip(self.buf_state.par_chunks_mut(7))
            .zip(self.rngs.par_iter_mut())
            .enumerate()
            .for_each(
                |(
                    i,
                    (
                        (((((((terminated, truncated), reward), car), wp_idx), step), scan), state),
                        rng,
                    ),
                )| {
                    if let Some(actions) = actions {
                        *step += 1;
                        for _ in 0..ds {
                            car.step(actions[i * 2], actions[i * 2 + 1], dt / (ds as f64));
                        }
                    }

                    *terminated = map.car_collides(car);
                    *truncated = *step >= max_steps;

                    let prev_idx = *wp_idx;
                    let (px, py) = map.position_to_pixels(car.x, car.y);
                    *wp_idx = map.skeleton.get_idx(px, py);

                    let mut delta = *wp_idx as f64 - prev_idx as f64;
                    if delta > n_wps as f64 / 2.0 {
                        delta -= n_wps as f64;
                    } else if delta < -(n_wps as f64 / 2.0) {
                        delta += n_wps as f64;
                    }
                    let d_steer_abs = if let Some(actions) = actions {
                        actions[i * 2].abs()
                    } else {
                        0.0
                    };
                    *reward = delta / n_wps as f64 * 100.0 * (1.0 + car.velocity.max(0.0) / 10.0)
                        - 0.1 * (-3.0 * map.edt(px, py)).exp()
                        - 0.05 * d_steer_abs
                        - if *terminated { 100.0 } else { 0.0 };

                    if *terminated || *truncated {
                        let ri = rng.random_range(0..n_wps);
                        let (px, py, th) = map.skeleton.get_point(ri);
                        *step = 0;
                        *car = Car {
                            x: px,
                            y: py,
                            theta: th,
                            velocity: 0.0,
                            steering: 0.0,
                            yaw_rate: 0.0,
                            slip_angle: 0.0,
                            omega_f: 0.0,
                            omega_r: 0.0,
                            params: CarParams::random(rng, dr_fract),
                        };
                        *wp_idx = ri;
                    }

                    let (sin_h, cos_h) = car.theta.sin_cos();
                    let lidar_x = car.x + car.params.lf * cos_h;
                    let lidar_y = car.y + car.params.lf * sin_h;
                    for (j, &(sin_a, cos_a)) in beam_sin_cos.iter().enumerate() {
                        let dx = cos_h * cos_a - sin_h * sin_a;
                        let dy = sin_h * cos_a + cos_h * sin_a;
                        let noise = rng.random_range(-0.03_f64..=0.03);
                        scan[j] = (map.raycast(lidar_x, lidar_y, dx, dy, max_range) + noise)
                            .clamp(min_range, max_range);
                    }
                    scan[n_beams] = car.velocity;
                    scan[n_beams + 1] = car.steering;
                    scan[n_beams + 2] = car.yaw_rate;

                    state[0] = car.x;
                    state[1] = car.y;
                    state[2] = car.theta;
                    state[3] = car.velocity;
                    state[4] = car.steering;
                    state[5] = car.yaw_rate;
                    state[6] = car.slip_angle;
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
