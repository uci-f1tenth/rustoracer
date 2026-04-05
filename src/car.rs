use rand::RngExt;
use rand::rngs::SmallRng;
use std::f64::consts::PI;

pub const G: f64 = 9.81;
pub const STEER_MIN: f64 = -0.5236;
pub const STEER_MAX: f64 = 0.5236;
pub const STEER_VEL_MIN: f64 = -3.2;
pub const STEER_VEL_MAX: f64 = 3.2;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.27;
pub const LENGTH: f64 = 0.50;

const CD_A_RHO_HALF: f64 = 0.5 * 0.3 * 0.04 * 1.225;
const V_S: f64 = 0.2;
const V_B: f64 = 0.05;
const V_MIN_DYN: f64 = 0.1;
const OMEGA_TAU: f64 = 0.02;

const DEF_LF: f64 = 0.15532;
const DEF_LR: f64 = 0.16868;
const DEF_H: f64 = 0.01434;
const DEF_MASS: f64 = 3.906;
const DEF_I_Z: f64 = 0.04712;
const DEF_V_SWITCH: f64 = 7.319;
const DEF_A_MAX: f64 = 9.51;
const DEF_R_W: f64 = 0.059;
const DEF_I_YW: f64 = 0.008;
const DEF_T_SB: f64 = 0.5;
const DEF_T_SE: f64 = 0.5;

type State = [f64; 9];

#[derive(Clone)]
pub struct TireParams {
    pub p_cx1: f64,
    pub p_dx1: f64,
    pub p_ex1: f64,
    pub p_kx1: f64,
    pub p_hx1: f64,
    pub p_vx1: f64,
    pub r_bx1: f64,
    pub r_bx2: f64,
    pub r_cx1: f64,
    pub r_ex1: f64,
    pub r_hx1: f64,
    pub p_cy1: f64,
    pub p_dy1: f64,
    pub p_ey1: f64,
    pub p_ky1: f64,
    pub r_by1: f64,
    pub r_by2: f64,
    pub r_by3: f64,
    pub r_cy1: f64,
    pub r_ey1: f64,
    pub r_hy1: f64,
    pub r_vy1: f64,
    pub r_vy4: f64,
    pub r_vy5: f64,
    pub r_vy6: f64,
}

impl TireParams {
    pub fn default() -> Self {
        Self {
            p_cx1: 1.6411,
            p_dx1: 1.1739,
            p_ex1: 0.4640,
            p_kx1: 5.42,
            p_hx1: 1.2297e-3,
            p_vx1: -8.8098e-6,
            r_bx1: 13.276,
            r_bx2: -13.778,
            r_cx1: 1.2568,
            r_ex1: 0.6522,
            r_hx1: 5.0722e-3,
            p_cy1: 1.3507,
            p_dy1: 1.0489,
            p_ey1: -7.4722e-3,
            p_ky1: -5.33,
            r_by1: 7.1433,
            r_by2: 9.1916,
            r_by3: -2.7856e-2,
            r_cy1: 1.0719,
            r_ey1: -0.2757,
            r_hy1: 5.7448e-6,
            r_vy1: -2.7825e-2,
            r_vy4: 12.120,
            r_vy5: 1.9,
            r_vy6: -10.704,
        }
    }
}

fn tire_fx_pure(s: f64, f_z: f64, tp: &TireParams) -> f64 {
    if f_z <= 0.0 {
        return 0.0;
    }
    let kx = -s + tp.p_hx1;
    let d_x = tp.p_dx1 * f_z;
    let b_x = if d_x.abs() > 1e-10 {
        f_z * tp.p_kx1 / (tp.p_cx1 * d_x)
    } else {
        0.0
    };
    let bkx = b_x * kx;
    d_x * (tp.p_cx1 * (bkx - tp.p_ex1 * (bkx - bkx.atan())).atan()).sin() + f_z * tp.p_vx1
}

fn tire_fy_pure(alpha: f64, f_z: f64, tp: &TireParams) -> (f64, f64) {
    if f_z <= 0.0 {
        return (0.0, tp.p_dy1);
    }
    let d_y = tp.p_dy1 * f_z;
    let b_y = if d_y.abs() > 1e-10 {
        f_z * tp.p_ky1 / (tp.p_cy1 * d_y)
    } else {
        0.0
    };
    let ba = b_y * alpha;
    (
        d_y * (tp.p_cy1 * (ba - tp.p_ey1 * (ba - ba.atan())).atan()).sin(),
        tp.p_dy1,
    )
}

fn tire_fx_combined(s: f64, alpha: f64, fx0: f64, tp: &TireParams) -> f64 {
    let kappa = -s;
    let b_xa = tp.r_bx1 * (tp.r_bx2 * kappa).atan().cos();
    let bs = b_xa * tp.r_hx1;
    let denom = (tp.r_cx1 * (bs - tp.r_ex1 * (bs - bs.atan())).atan()).cos();
    let d_xa = if denom.abs() > 1e-10 {
        fx0 / denom
    } else {
        fx0
    };
    let ba = b_xa * (alpha + tp.r_hx1);
    d_xa * (tp.r_cx1 * (ba - tp.r_ex1 * (ba - ba.atan())).atan()).cos()
}

fn tire_fy_combined(s: f64, alpha: f64, mu_y: f64, f_z: f64, fy0: f64, tp: &TireParams) -> f64 {
    let kappa = -s;
    let kappa_s = kappa + tp.r_hy1;
    let b_yk = tp.r_by1 * (tp.r_by2 * (alpha - tp.r_by3)).atan().cos();
    let bsh = b_yk * tp.r_hy1;
    let denom = (tp.r_cy1 * (bsh - tp.r_ey1 * (bsh - bsh.atan())).atan()).cos();
    let d_yk = if denom.abs() > 1e-10 {
        fy0 / denom
    } else {
        fy0
    };
    let d_vyk = mu_y * f_z * tp.r_vy1 * (tp.r_vy4 * alpha).atan().cos();
    let s_vyk = d_vyk * (tp.r_vy5 * (tp.r_vy6 * kappa).atan()).sin();
    let bk = b_yk * kappa_s;
    d_yk * (tp.r_cy1 * (bk - tp.r_ey1 * (bk - bk.atan())).atan()).cos() + s_vyk
}

#[derive(Clone)]
pub struct CarParams {
    pub lf: f64,
    pub lr: f64,
    pub h: f64,
    pub mass: f64,
    pub i_z: f64,
    pub v_switch: f64,
    pub a_max: f64,
    pub r_w: f64,
    pub i_yw: f64,
    pub t_sb: f64,
    pub t_se: f64,
    pub tire: TireParams,
}

impl CarParams {
    pub fn default() -> Self {
        Self {
            lf: DEF_LF,
            lr: DEF_LR,
            h: DEF_H,
            mass: DEF_MASS,
            i_z: DEF_I_Z,
            v_switch: DEF_V_SWITCH,
            a_max: DEF_A_MAX,
            r_w: DEF_R_W,
            i_yw: DEF_I_YW,
            t_sb: DEF_T_SB,
            t_se: DEF_T_SE,
            tire: TireParams::default(),
        }
    }

    pub fn random(rng: &mut SmallRng, frac: f64) -> Self {
        let r = |rng: &mut SmallRng, v: f64| v * rng.random_range(1.0 - frac..=1.0 + frac);
        let mut p = Self::default();
        p.lf = r(rng, DEF_LF);
        p.lr = r(rng, DEF_LR);
        p.h = r(rng, DEF_H);
        p.mass = r(rng, DEF_MASS);
        p.i_z = r(rng, DEF_I_Z);
        p.v_switch = r(rng, DEF_V_SWITCH);
        p.a_max = r(rng, DEF_A_MAX);
        p.r_w = r(rng, DEF_R_W);
        p.i_yw = r(rng, DEF_I_YW);
        let mu = rng.random_range(1.0 - frac..=1.0 + frac);
        p.tire.p_dy1 *= mu;
        p.tire.p_dx1 *= mu;
        let st = rng.random_range(1.0 - frac..=1.0 + frac);
        p.tire.p_ky1 *= st;
        p.tire.p_kx1 *= st;
        p
    }

    #[inline]
    pub fn lwb(&self) -> f64 {
        self.lf + self.lr
    }
}

fn steering_constraint(steer: f64, sv: f64) -> f64 {
    let sv = sv.clamp(STEER_VEL_MIN, STEER_VEL_MAX);
    if sv < 0.0 && steer <= STEER_MIN {
        0.0
    } else if sv > 0.0 && steer >= STEER_MAX {
        0.0
    } else {
        sv
    }
}

fn std_dynamics(s: &State, torque: f64, sv: f64, p: &CarParams) -> State {
    let [_, _, delta, v, _psi, psi_dot, beta, omega_f, omega_r] = *s;
    let lwb = p.lwb();
    let tp = &p.tire;
    let omega_f = omega_f.max(0.0);
    let omega_r = omega_r.max(0.0);

    let (alpha_f, alpha_r) = if v > V_MIN_DYN {
        let vc = v * beta.cos();
        let vs = v * beta.sin();
        (
            ((vs + psi_dot * p.lf) / vc).atan() - delta,
            ((vs - psi_dot * p.lr) / vc).atan(),
        )
    } else {
        (0.0, 0.0)
    };

    let a_est = torque / (p.mass * p.r_w);
    let f_zf = p.mass * (-a_est * p.h + G * p.lr) / lwb;
    let f_zr = p.mass * (a_est * p.h + G * p.lf) / lwb;

    let u_wf =
        (v * beta.cos() * delta.cos() + (v * beta.sin() + p.lf * psi_dot) * delta.sin()).max(0.0);
    let u_wr = (v * beta.cos()).max(0.0);

    let s_f = (1.0 - p.r_w * omega_f / u_wf.max(V_MIN_DYN)).clamp(-1.0, 1.0);
    let s_r = (1.0 - p.r_w * omega_r / u_wr.max(V_MIN_DYN)).clamp(-1.0, 1.0);

    let fx0_f = tire_fx_pure(s_f, f_zf, tp);
    let fx0_r = tire_fx_pure(s_r, f_zr, tp);
    let (fy0_f, mu_yf) = tire_fy_pure(alpha_f, f_zf, tp);
    let (fy0_r, mu_yr) = tire_fy_pure(alpha_r, f_zr, tp);

    let f_xf = tire_fx_combined(s_f, alpha_f, fx0_f, tp);
    let f_xr = tire_fx_combined(s_r, alpha_r, fx0_r, tp);
    let f_yf = tire_fy_combined(s_f, alpha_f, mu_yf, f_zf, fy0_f, tp);
    let f_yr = tire_fy_combined(s_r, alpha_r, mu_yr, f_zr, fy0_r, tp);

    let (t_b, t_e) = if torque > 0.0 {
        (0.0, torque)
    } else {
        (torque, 0.0)
    };
    let f_drag = CD_A_RHO_HALF * v * v.abs();

    let d_v_dyn = (1.0 / p.mass)
        * (-f_yf * (delta - beta).sin()
            + f_yr * beta.sin()
            + f_xr * beta.cos()
            + f_xf * (delta - beta).cos()
            - f_drag);

    let dd_psi_dyn =
        (1.0 / p.i_z) * (f_yf * delta.cos() * p.lf - f_yr * p.lr + f_xf * delta.sin() * p.lf);

    let d_beta_dyn = if v > V_MIN_DYN {
        -psi_dot
            + (1.0 / (p.mass * v))
                * (f_yf * (delta - beta).cos() + f_yr * beta.cos() - f_xr * beta.sin()
                    + f_xf * (delta - beta).sin())
    } else {
        0.0
    };

    let wheel_dyn = |omega: f64, f_x: f64, t_frac_b: f64, t_frac_e: f64| {
        let net = -p.r_w * f_x + t_frac_b * t_b + t_frac_e * t_e;
        if omega > 0.0 {
            net / p.i_yw
        } else {
            net.max(0.0) / p.i_yw
        }
    };
    let d_omega_f_dyn = wheel_dyn(omega_f, f_xf, p.t_sb, p.t_se);
    let d_omega_r_dyn = wheel_dyn(omega_r, f_xr, 1.0 - p.t_sb, 1.0 - p.t_se);

    let d_v_ks = a_est - f_drag / p.mass;
    let d_psi_ks = v * beta.cos() / lwb * delta.tan();
    let cos2_d = delta.cos().powi(2);
    let tan_lr_lwb = delta.tan() * p.lr / lwb;

    let d_beta_ks = if cos2_d.abs() > 1e-10 {
        (p.lr * sv) / (lwb * cos2_d * (1.0 + tan_lr_lwb * tan_lr_lwb))
    } else {
        0.0
    };

    let dd_psi_ks = if cos2_d.abs() > 1e-10 {
        (1.0 / lwb)
            * (a_est * beta.cos() * delta.tan() - v * beta.sin() * d_beta_ks * delta.tan()
                + v * beta.cos() / cos2_d * sv)
    } else {
        0.0
    };

    let d_omega_f_ks = (u_wf / p.r_w - omega_f) / OMEGA_TAU;
    let d_omega_r_ks = (u_wr / p.r_w - omega_r) / OMEGA_TAU;

    let w_dyn = 0.5 * ((v - V_S) / V_B).tanh() + 0.5;
    let w_ks = 1.0 - w_dyn;

    let psi = s[4];
    [
        v * (beta + psi).cos(),
        v * (beta + psi).sin(),
        sv,
        w_dyn * d_v_dyn + w_ks * d_v_ks,
        w_dyn * psi_dot + w_ks * d_psi_ks,
        w_dyn * dd_psi_dyn + w_ks * dd_psi_ks,
        w_dyn * d_beta_dyn + w_ks * d_beta_ks,
        w_dyn * d_omega_f_dyn + w_ks * d_omega_f_ks,
        w_dyn * d_omega_r_dyn + w_ks * d_omega_r_ks,
    ]
}

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&std::array::from_fn(|i| y[i] + k1[i] * dt / 2.0));
    let k3 = f(&std::array::from_fn(|i| y[i] + k2[i] * dt / 2.0));
    let k4 = f(&std::array::from_fn(|i| y[i] + k3[i] * dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
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
    pub omega_f: f64,
    pub omega_r: f64,
    pub params: CarParams,
}

impl Car {
    pub fn new(x: f64, y: f64, theta: f64, velocity: f64, params: CarParams) -> Self {
        let omega = if velocity > 0.0 {
            velocity / params.r_w
        } else {
            0.0
        };
        Self {
            x,
            y,
            theta,
            velocity,
            steering: 0.0,
            yaw_rate: 0.0,
            slip_angle: 0.0,
            omega_f: omega,
            omega_r: omega,
            params,
        }
    }

    pub fn step(&mut self, d_steer: f64, torque: f64, dt: f64) {
        let p = &self.params;
        let sv = steering_constraint(self.steering, d_steer * STEER_VEL_MAX);
        let torque = torque.clamp(-1.0, 1.0) * p.mass * p.a_max * p.r_w;

        let state: State = [
            self.x,
            self.y,
            self.steering,
            self.velocity,
            self.theta,
            self.yaw_rate,
            self.slip_angle,
            self.omega_f,
            self.omega_r,
        ];
        let [x, y, s, vx, yaw, yr, slip, wf, wr] =
            rk4(&state, dt, |s| std_dynamics(s, torque, sv, p));

        self.x = x;
        self.y = y;
        self.steering = s.clamp(STEER_MIN, STEER_MAX);
        self.velocity = vx.clamp(V_MIN, V_MAX);
        self.theta = (yaw + PI).rem_euclid(2.0 * PI) - PI;
        self.yaw_rate = yr;
        self.slip_angle = slip;
        self.omega_f = wf.max(0.0);
        self.omega_r = wr.max(0.0);
    }
}
