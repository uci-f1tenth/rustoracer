use rand::RngExt;
use rand::rngs::SmallRng;
use std::f64::consts::PI;

pub const G: f64 = 9.81;

// ── Constraint limits ────────────────────────────────────────────────────────

pub const STEER_MIN: f64 = -0.5236;
pub const STEER_MAX: f64 = 0.5236;
pub const STEER_VEL_MIN: f64 = -3.2;
pub const STEER_VEL_MAX: f64 = 3.2;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.27;
pub const LENGTH: f64 = 0.50;
const CD_A_RHO_HALF: f64 = 0.5 * 0.3 * 0.04 * 1.225; // aero drag 0.5·Cd·A·ρ

// ── Model blending parameters (smooth kinematic↔dynamic transition) ──────────
// Reference: CommonRoad STD Python implementation

const V_S: f64 = 0.2; // switching velocity
const V_B: f64 = 0.05; // blending bandwidth (tanh steepness)
const V_MIN_DYN: f64 = 0.1; // minimum velocity for dynamic computations (= V_S/2)
const OMEGA_TAU: f64 = 0.02; // time constant for kinematic wheel speed tracking [s]

// ── Default vehicle parameters (F1Tenth scale) ──────────────────────────────
//
// The original ST model used linear cornering stiffness coefficients:
//   C_S,f = 4.718    C_S,r = 5.4562    µ = 1.0489
//
// In the CommonRoad Pacejka model the equivalent relationship is:
//   C_S = |p_Ky1| / p_Dy1     (Section 10.1)
//
// Full-scale CommonRoad tires have C_S = 21.92 / 1.0489 ≈ 20.89.
// F1Tenth tires are much less stiff, so we scale p_Ky1 and p_Kx1 to match:
//   p_Ky1 = -(avg C_S) * p_Dy1 = -5.08 * 1.0489 ≈ -5.33
//   p_Kx1 scaled by the same ratio:  22.303 * (5.08 / 20.89) ≈ 5.42
//
// Wheel inertia includes the effective drivetrain inertia (motor rotor,
// gears, belts). The bare wheel (~50g at r=32mm) gives I ≈ 5e-5 kg·m²,
// but the drivetrain reflected inertia dominates. A value of 0.004 keeps
// the wheel dynamics time constant at ~5–10 ms, which is well-resolved
// by RK4 at typical simulation timesteps (1–10 ms).

const DEF_LF: f64 = 0.15532;
const DEF_LR: f64 = 0.16868;
const DEF_H: f64 = 0.01434;
const DEF_MASS: f64 = 3.906;
const DEF_I_Z: f64 = 0.04712;
const DEF_V_SWITCH: f64 = 7.319;
const DEF_A_MAX: f64 = 9.51;
const DEF_R_W: f64 = 0.059; // effective tire radius [m]
const DEF_I_YW: f64 = 0.008; // effective wheel + drivetrain inertia [kg·m²]
const DEF_T_SB: f64 = 0.5; // brake torque split to front axle
const DEF_T_SE: f64 = 0.5; // engine torque split to front axle (0 = RWD)

/// State vector: [sx, sy, δ, v, ψ, ψ̇, β, ωf, ωr]
type State = [f64; 9];

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Pacejka 2002 Tire Model
//  Reference: CommonRoad Vehicle Models v2020a, Section 9.3
//             MSC Adams/Tire PAC2002
//
//  Shape factors (p_Cx1, p_Cy1, p_Ex1, p_Ey1, etc.) are dimensionless
//  curve-shape parameters that transfer across scales. Stiffness parameters
//  (p_Kx1, p_Ky1) and friction (p_Dx1, p_Dy1) are scaled for F1Tenth tires.
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct TireParams {
    // ── Longitudinal pure slip ──
    pub p_cx1: f64,
    pub p_dx1: f64,
    pub p_dx3: f64,
    pub p_ex1: f64,
    pub p_kx1: f64,
    pub p_hx1: f64,
    pub p_vx1: f64,

    // ── Longitudinal combined slip ──
    pub r_bx1: f64,
    pub r_bx2: f64,
    pub r_cx1: f64,
    pub r_ex1: f64,
    pub r_hx1: f64,

    // ── Lateral pure slip ──
    pub p_cy1: f64,
    pub p_dy1: f64,
    pub p_dy3: f64,
    pub p_ey1: f64,
    pub p_ky1: f64,
    pub p_hy1: f64,
    pub p_hy3: f64,
    pub p_vy1: f64,
    pub p_vy3: f64,

    // ── Lateral combined slip ──
    pub r_by1: f64,
    pub r_by2: f64,
    pub r_by3: f64,
    pub r_cy1: f64,
    pub r_ey1: f64,
    pub r_hy1: f64,
    pub r_vy1: f64,
    pub r_vy3: f64,
    pub r_vy4: f64,
    pub r_vy5: f64,
    pub r_vy6: f64,
}

impl TireParams {
    /// Default Pacejka parameters scaled for F1Tenth tires.
    ///
    /// Shape factors are from CommonRoad Table 7 (PAC2002).
    /// Stiffness values (p_kx1, p_ky1) are scaled down to match
    /// the empirical F1Tenth cornering stiffness coefficients
    /// (C_S,f ≈ 4.718, C_S,r ≈ 5.456 from the original ST model).
    pub fn default() -> Self {
        Self {
            // ── Longitudinal pure slip ──
            p_cx1: 1.6411,
            p_dx1: 1.1739,
            p_dx3: 0.0,
            p_ex1: 0.4640,
            p_kx1: 5.42, // scaled from 22.303 for F1Tenth
            p_hx1: 1.2297e-3,
            p_vx1: -8.8098e-6,

            // ── Longitudinal combined slip ──
            r_bx1: 13.276,
            r_bx2: -13.778,
            r_cx1: 1.2568,
            r_ex1: 0.6522,
            r_hx1: 5.0722e-3,

            // ── Lateral pure slip ──
            p_cy1: 1.3507,
            p_dy1: 1.0489, // µ_y — matches original DEF_MU
            p_dy3: -2.8821,
            p_ey1: -7.4722e-3,
            p_ky1: -5.33, // scaled from -21.920 for F1Tenth
            p_hy1: 2.6747e-3,
            p_hy3: 3.1415e-2,
            p_vy1: 3.7318e-2,
            p_vy3: -0.3293,

            // ── Lateral combined slip ──
            r_by1: 7.1433,
            r_by2: 9.1916,
            r_by3: -2.7856e-2,
            r_cy1: 1.0719,
            r_ey1: -0.2757,
            r_hy1: 5.7448e-6,
            r_vy1: -2.7825e-2,
            r_vy3: -0.2756,
            r_vy4: 12.120,
            r_vy5: 1.9,
            r_vy6: -10.704,
        }
    }
}

// ── Pure slip longitudinal force (PAC2002 eq. 18–28) ─────────────────────────

fn tire_fx_pure(s: f64, gamma: f64, f_z: f64, tp: &TireParams) -> f64 {
    if f_z <= 0.0 {
        return 0.0;
    }
    let kappa = -s;
    let kx = kappa + tp.p_hx1;
    let mu_x = tp.p_dx1 * (1.0 - tp.p_dx3 * gamma * gamma);
    let c_x = tp.p_cx1;
    let d_x = mu_x * f_z;
    let e_x = tp.p_ex1;
    let k_x = f_z * tp.p_kx1;
    let b_x = if d_x.abs() > 1e-10 {
        k_x / (c_x * d_x)
    } else {
        0.0
    };

    let bkx = b_x * kx;
    d_x * (c_x * (bkx - e_x * (bkx - bkx.atan())).atan()).sin() + f_z * tp.p_vx1
}

// ── Pure slip lateral force (PAC2002 eq. 30–41) → (Fy0, µy) ─────────────────

fn tire_fy_pure(alpha: f64, gamma: f64, f_z: f64, tp: &TireParams) -> (f64, f64) {
    let mu_y = tp.p_dy1 * (1.0 - tp.p_dy3 * gamma * gamma);
    if f_z <= 0.0 {
        return (0.0, mu_y);
    }
    let g_sgn = sign(gamma);
    let g_abs = gamma.abs();
    let s_hy = g_sgn * (tp.p_hy1 + tp.p_hy3 * g_abs);
    let s_vy = g_sgn * f_z * (tp.p_vy1 + tp.p_vy3 * g_abs);
    let ay = alpha + s_hy;
    let c_y = tp.p_cy1;
    let d_y = mu_y * f_z;
    let e_y = tp.p_ey1;
    let k_y = f_z * tp.p_ky1;
    let b_y = if d_y.abs() > 1e-10 {
        k_y / (c_y * d_y)
    } else {
        0.0
    };

    let bay = b_y * ay;
    let fy0 = d_y * (c_y * (bay - e_y * (bay - bay.atan())).atan()).sin() + s_vy;
    (fy0, mu_y)
}

// ── Combined slip longitudinal force (PAC2002 eq. 59–65) ─────────────────────

fn tire_fx_combined(s: f64, alpha: f64, fx0: f64, tp: &TireParams) -> f64 {
    let kappa = -s;
    let s_hxa = tp.r_hx1;
    let alpha_s = alpha + s_hxa;
    let b_xa = tp.r_bx1 * (tp.r_bx2 * kappa).atan().cos();
    let c_xa = tp.r_cx1;
    let e_xa = tp.r_ex1;

    let bs = b_xa * s_hxa;
    let denom = (c_xa * (bs - e_xa * (bs - bs.atan())).atan()).cos();
    let d_xa = if denom.abs() > 1e-10 {
        fx0 / denom
    } else {
        fx0
    };

    let ba = b_xa * alpha_s;
    d_xa * (c_xa * (ba - e_xa * (ba - ba.atan())).atan()).cos()
}

// ── Combined slip lateral force (PAC2002 eq. 68–76) ──────────────────────────

fn tire_fy_combined(
    s: f64,
    alpha: f64,
    gamma: f64,
    mu_y: f64,
    f_z: f64,
    fy0: f64,
    tp: &TireParams,
) -> f64 {
    let kappa = -s;
    let s_hyk = tp.r_hy1;
    let kappa_s = kappa + s_hyk;
    let b_yk = tp.r_by1 * (tp.r_by2 * (alpha - tp.r_by3)).atan().cos();
    let c_yk = tp.r_cy1;
    let e_yk = tp.r_ey1;

    let bsh = b_yk * s_hyk;
    let denom = (c_yk * (bsh - e_yk * (bsh - bsh.atan())).atan()).cos();
    let d_yk = if denom.abs() > 1e-10 {
        fy0 / denom
    } else {
        fy0
    };

    let d_vyk = mu_y * f_z * (tp.r_vy1 + tp.r_vy3 * gamma) * (tp.r_vy4 * alpha).atan().cos();
    let s_vyk = d_vyk * (tp.r_vy5 * (tp.r_vy6 * kappa).atan()).sin();

    let bk = b_yk * kappa_s;
    d_yk * (c_yk * (bk - e_yk * (bk - bk.atan())).atan()).cos() + s_vyk
}

// ══════════════════════════════════════════════════════════════════════════════
//  Vehicle Parameters
// ══════════════════════════════════════════════════════════════════════════════

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

        // Scale both friction coefficients together
        let mu_scale = rng.random_range(1.0 - frac..=1.0 + frac);
        p.tire.p_dy1 *= mu_scale;
        p.tire.p_dx1 *= mu_scale;

        // Scale tire stiffness (cornering + longitudinal)
        let stiff_scale = rng.random_range(1.0 - frac..=1.0 + frac);
        p.tire.p_ky1 *= stiff_scale;
        p.tire.p_kx1 *= stiff_scale;

        p
    }

    #[inline]
    pub fn lwb(&self) -> f64 {
        self.lf + self.lr
    }
}

// ── Constraint functions (CommonRoad eq. 4–5) ────────────────────────────────

fn steering_constraint(steer: f64, sv: f64) -> f64 {
    let sv = sv.clamp(STEER_VEL_MIN, STEER_VEL_MAX);
    match sv {
        sv if sv < 0.0 && steer <= STEER_MIN => 0.0,
        sv if sv > 0.0 && steer >= STEER_MAX => 0.0,
        sv => sv,
    }
}

// ── Single-Track Drift dynamics (CommonRoad §8, eq. 14–16) ───────────────────
// Smoothly blended with kinematic model at low speeds via tanh weighting.

fn std_dynamics(s: &State, torque: f64, sv: f64, p: &CarParams) -> State {
    let [_sx, _sy, delta, v, psi, psi_dot, beta, omega_f, omega_r] = *s;
    let lwb = p.lwb();
    let tp = &p.tire;

    let omega_f = omega_f.max(0.0);
    let omega_r = omega_r.max(0.0);
    let gamma: f64 = 0.0; // camber = 0 for single-track

    // ── Lateral slip angles ──────────────────────────────────────────────
    let (alpha_f, alpha_r) = if v > V_MIN_DYN {
        let v_cos_b = v * beta.cos();
        let v_sin_b = v * beta.sin();
        (
            ((v_sin_b + psi_dot * p.lf) / v_cos_b).atan() - delta,
            ((v_sin_b - psi_dot * p.lr) / v_cos_b).atan(),
        )
    } else {
        (0.0, 0.0)
    };

    // ── Estimated acceleration (for load transfer & kinematic model) ─────
    let a_est = torque / (p.mass * p.r_w);

    // ── Vertical tire forces (load transfer) ─────────────────────────────
    let f_zf = p.mass * (-a_est * p.h + G * p.lr) / lwb;
    let f_zr = p.mass * (a_est * p.h + G * p.lf) / lwb;

    // ── Individual tire velocities ───────────────────────────────────────
    let u_wf =
        (v * beta.cos() * delta.cos() + (v * beta.sin() + p.lf * psi_dot) * delta.sin()).max(0.0);
    let u_wr = (v * beta.cos()).max(0.0);

    // ── Longitudinal slip (clamped for numerical stability) ──────────────
    let s_f = (1.0 - p.r_w * omega_f / u_wf.max(V_MIN_DYN)).clamp(-1.0, 1.0);
    let s_r = (1.0 - p.r_w * omega_r / u_wr.max(V_MIN_DYN)).clamp(-1.0, 1.0);

    // ── Pacejka tire forces ──────────────────────────────────────────────
    let fx0_f = tire_fx_pure(s_f, gamma, f_zf, tp);
    let fx0_r = tire_fx_pure(s_r, gamma, f_zr, tp);
    let (fy0_f, mu_yf) = tire_fy_pure(alpha_f, gamma, f_zf, tp);
    let (fy0_r, mu_yr) = tire_fy_pure(alpha_r, gamma, f_zr, tp);

    let f_xf = tire_fx_combined(s_f, alpha_f, fx0_f, tp);
    let f_xr = tire_fx_combined(s_r, alpha_r, fx0_r, tp);
    let f_yf = tire_fy_combined(s_f, alpha_f, gamma, mu_yf, f_zf, fy0_f, tp);
    let f_yr = tire_fy_combined(s_r, alpha_r, gamma, mu_yr, f_zr, fy0_r, tp);

    // ── Torque split → brake / engine (eq. 21) ──────────────────────────
    let (t_b, t_e) = if torque > 0.0 {
        (0.0, torque)
    } else {
        (torque, 0.0)
    };

    // ── Aerodynamic drag ─────────────────────────────────────────────────
    let f_drag = CD_A_RHO_HALF * v * v * sign(v);

    // ══════════════════════════════════════════════════════════════════════
    //  DYNAMIC MODEL (eq. 14–15)
    // ══════════════════════════════════════════════════════════════════════

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

    let d_omega_f_dyn = if omega_f > 0.0 {
        (1.0 / p.i_yw) * (-p.r_w * f_xf + p.t_sb * t_b + p.t_se * t_e)
    } else {
        // Allow spin-up from rest when torque is positive
        let net_torque = -p.r_w * f_xf + p.t_sb * t_b + p.t_se * t_e;
        if net_torque > 0.0 {
            net_torque / p.i_yw
        } else {
            0.0
        }
    };
    let d_omega_r_dyn = if omega_r > 0.0 {
        (1.0 / p.i_yw) * (-p.r_w * f_xr + (1.0 - p.t_sb) * t_b + (1.0 - p.t_se) * t_e)
    } else {
        let net_torque = -p.r_w * f_xr + (1.0 - p.t_sb) * t_b + (1.0 - p.t_se) * t_e;
        if net_torque > 0.0 {
            net_torque / p.i_yw
        } else {
            0.0
        }
    };

    // ══════════════════════════════════════════════════════════════════════
    //  KINEMATIC MODEL at COG (eq. 13)
    // ══════════════════════════════════════════════════════════════════════

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

    // Kinematic wheel speed tracking (first-order filter → no-slip speed)
    let d_omega_f_ks = (1.0 / OMEGA_TAU) * (u_wf / p.r_w - omega_f);
    let d_omega_r_ks = (1.0 / OMEGA_TAU) * (u_wr / p.r_w - omega_r);

    // ══════════════════════════════════════════════════════════════════════
    //  SMOOTH BLENDING
    // ══════════════════════════════════════════════════════════════════════

    let w_dyn = 0.5 * ((v - V_S) / V_B).tanh() + 0.5;
    let w_ks = 1.0 - w_dyn;

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

// ── RK4 integrator ───────────────────────────────────────────────────────────

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&std::array::from_fn(|i| y[i] + k1[i] * dt / 2.0));
    let k3 = f(&std::array::from_fn(|i| y[i] + k2[i] * dt / 2.0));
    let k4 = f(&std::array::from_fn(|i| y[i] + k3[i] * dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

// ══════════════════════════════════════════════════════════════════════════════
//  Car
// ══════════════════════════════════════════════════════════════════════════════

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

        // ── Steering: interpret d_steer [-1, 1] as normalized steering velocity
        let sv = steering_constraint(self.steering, d_steer * STEER_VEL_MAX);

        // ── Rescale normalized torque [-1, 1] to physical units ─────────
        let t_max = p.mass * p.a_max * p.r_w;
        let torque = torque.clamp(-1.0, 1.0) * t_max;

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
