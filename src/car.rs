use rand::RngExt;
use rand::rngs::SmallRng;

pub const G: f64 = 9.81;

const DEF_MU: f64 = 1.0489;
const DEF_C_SF: f64 = 4.718;
const DEF_C_SR: f64 = 5.4562;
const DEF_LF: f64 = 0.15875;
const DEF_LR: f64 = 0.17145;
const DEF_H: f64 = 0.074;
const DEF_MASS: f64 = 3.74;
const DEF_I_Z: f64 = 0.04712;
const DEF_V_SWITCH: f64 = 7.319;
const DEF_A_MAX: f64 = 9.51;

pub const STEER_MIN: f64 = -0.4189;
pub const STEER_MAX: f64 = 0.4189;
pub const STEER_VEL_MIN: f64 = -3.2;
pub const STEER_VEL_MAX: f64 = 3.2;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.31;
pub const LENGTH: f64 = 0.58;

const V_KIN_THRESHOLD: f64 = 0.5;

type State = [f64; 7];

#[derive(Clone)]
pub struct CarParams {
    pub mu: f64,
    pub c_sf: f64,
    pub c_sr: f64,
    pub lf: f64,
    pub lr: f64,
    pub h: f64,
    pub mass: f64,
    pub i_z: f64,
    pub v_switch: f64,
    pub a_max: f64,
}

impl CarParams {
    pub fn default() -> Self {
        Self {
            mu: DEF_MU,
            c_sf: DEF_C_SF,
            c_sr: DEF_C_SR,
            lf: DEF_LF,
            lr: DEF_LR,
            h: DEF_H,
            mass: DEF_MASS,
            i_z: DEF_I_Z,
            v_switch: DEF_V_SWITCH,
            a_max: DEF_A_MAX,
        }
    }

    pub fn random(rng: &mut SmallRng, frac: f64) -> Self {
        let r = |rng: &mut SmallRng, v: f64| v * rng.random_range(1.0 - frac..=1.0 + frac);
        Self {
            mu: r(rng, DEF_MU),
            c_sf: r(rng, DEF_C_SF),
            c_sr: r(rng, DEF_C_SR),
            lf: r(rng, DEF_LF),
            lr: r(rng, DEF_LR),
            h: r(rng, DEF_H),
            mass: r(rng, DEF_MASS),
            i_z: r(rng, DEF_I_Z),
            v_switch: r(rng, DEF_V_SWITCH),
            a_max: r(rng, DEF_A_MAX),
        }
    }

    #[inline]
    pub fn lwb(&self) -> f64 {
        self.lf + self.lr
    }
}

#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub velocity: f64,
    pub steering: f64,
    pub yaw_rate: f64,
    pub slip_angle: f64,
    pub params: CarParams,
}

fn steering_constraint(steer: f64, sv: f64) -> f64 {
    let sv = sv.clamp(STEER_VEL_MIN, STEER_VEL_MAX);
    match sv {
        sv if sv < 0.0 && steer <= STEER_MIN => 0.0,
        sv if sv > 0.0 && steer >= STEER_MAX => 0.0,
        sv => sv,
    }
}

fn accel_constraint(vel: f64, a: f64, p: &CarParams) -> f64 {
    let a_max = if vel > p.v_switch {
        p.a_max * p.v_switch / vel
    } else {
        p.a_max
    };
    let a = a.clamp(-p.a_max, a_max);
    match a {
        a if a < 0.0 && vel <= V_MIN => 0.0,
        a if a > 0.0 && vel >= V_MAX => 0.0,
        a => a,
    }
}

fn kinematic_dynamics(s: &State, a: f64, sv: f64, p: &CarParams) -> State {
    let [_, _, steer, vx, yaw, _, _] = *s;
    let lwb = p.lwb();
    let beta = (p.lr / lwb * steer.tan()).atan();
    let (sin_b, cos_sq) = (beta.sin(), steer.cos().powi(2));
    [
        vx * (yaw + beta).cos(),
        vx * (yaw + beta).sin(),
        sv,
        a,
        vx / p.lr * sin_b,
        a / lwb * steer.tan() + vx / (lwb * cos_sq) * sv,
        0.0,
    ]
}

fn dynamic_dynamics(s: &State, a: f64, sv: f64, p: &CarParams) -> State {
    let [_, _, steer, vx, yaw, yr, slip] = *s;
    let lwb = p.lwb();
    let vx_safe = if vx >= 0.0 {
        vx.max(V_KIN_THRESHOLD)
    } else {
        vx.min(-V_KIN_THRESHOLD)
    };

    let rear_load = G * p.lf + a * p.h;
    let front_load = G * p.lr - a * p.h;

    let yaw_acc = -p.mu * p.mass / (vx_safe * p.i_z * lwb)
        * (p.lf * p.lf * p.c_sf * front_load + p.lr * p.lr * p.c_sr * rear_load)
        * yr
        + p.mu * p.mass / (p.i_z * lwb)
            * (p.lr * p.c_sr * rear_load - p.lf * p.c_sf * front_load)
            * slip
        + p.mu * p.mass / (p.i_z * lwb) * p.lf * p.c_sf * front_load * steer;

    let slip_rate = (p.mu / (vx_safe * vx_safe * lwb)
        * (p.c_sr * rear_load * p.lr - p.c_sf * front_load * p.lf)
        - 1.0)
        * yr
        - p.mu / (vx_safe * lwb) * (p.c_sr * rear_load + p.c_sf * front_load) * slip
        + p.mu / (vx_safe * lwb) * p.c_sf * front_load * steer;

    [
        vx * (slip + yaw).cos(),
        vx * (slip + yaw).sin(),
        sv,
        a,
        yr,
        yaw_acc,
        slip_rate,
    ]
}

fn dynamics(s: &State, a: f64, sv: f64, p: &CarParams) -> State {
    if s[3].abs() < V_KIN_THRESHOLD {
        kinematic_dynamics(s, a, sv, p)
    } else {
        dynamic_dynamics(s, a, sv, p)
    }
}

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&std::array::from_fn(|i| y[i] + k1[i] * dt / 2.0));
    let k3 = f(&std::array::from_fn(|i| y[i] + k2[i] * dt / 2.0));
    let k4 = f(&std::array::from_fn(|i| y[i] + k3[i] * dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn pid(target_steer: f64, target_speed: f64, vx: f64, steer: f64, p: &CarParams) -> (f64, f64) {
    let sv = if (target_steer - steer).abs() > 1e-4 {
        (target_steer - steer).signum() * STEER_VEL_MAX
    } else {
        0.0
    };
    let kp = if vx > 0.0 { 10.0 } else { 2.0 } * p.a_max
        / if target_speed > vx { V_MAX } else { -V_MIN };
    (kp * (target_speed - vx), sv)
}

impl Car {
    pub fn step(&mut self, steer: f64, speed: f64, dt: f64) {
        let p = &self.params;
        let (raw_a, raw_sv) = pid(steer, speed, self.velocity, self.steering, p);
        let a = accel_constraint(self.velocity, raw_a, p);
        let sv = steering_constraint(self.steering, raw_sv);

        let state: State = [
            self.x,
            self.y,
            self.steering,
            self.velocity,
            self.theta,
            self.yaw_rate,
            self.slip_angle,
        ];
        let [x, y, s, vx, yaw, yr, slip] = rk4(&state, dt, |s| dynamics(s, a, sv, p));

        self.x = x;
        self.y = y;
        self.steering = s.clamp(STEER_MIN, STEER_MAX);
        self.velocity = vx.clamp(V_MIN, V_MAX);
        self.theta = yaw;
        self.yaw_rate = yr;
        self.slip_angle = slip;
    }
}
