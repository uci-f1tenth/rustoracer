pub const G: f64 = 9.81;
pub const MU: f64 = 1.0489;
pub const C_SF: f64 = 4.718;
pub const C_SR: f64 = 5.4562;
pub const LF: f64 = 0.15875;
pub const LR: f64 = 0.17145;
pub const LWB: f64 = LF + LR;
pub const H: f64 = 0.074;
pub const MASS: f64 = 3.74;
pub const I_Z: f64 = 0.04712;
pub const STEER_MIN: f64 = -0.4189;
pub const STEER_MAX: f64 = 0.4189;
pub const STEER_VEL_MIN: f64 = -3.2;
pub const STEER_VEL_MAX: f64 = 3.2;
pub const V_SWITCH: f64 = 7.319;
pub const A_MAX: f64 = 9.51;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.31;
pub const LENGTH: f64 = 0.58;

const V_KIN_THRESHOLD: f64 = 0.5;

/// State: [x, y, δ, vx, ψ, ψ̇, β]
type State = [f64; 7];

#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,      // yaw angle ψ
    pub velocity: f64,   // longitudinal velocity vx
    pub steering: f64,   // front wheel angle δ
    pub yaw_rate: f64,   // ψ̇
    pub slip_angle: f64, // β
}

fn steering_constraint(steer: f64, sv: f64) -> f64 {
    let sv = sv.clamp(STEER_VEL_MIN, STEER_VEL_MAX);
    match sv {
        sv if sv < 0.0 && steer <= STEER_MIN => 0.0,
        sv if sv > 0.0 && steer >= STEER_MAX => 0.0,
        sv => sv,
    }
}

fn accel_constraint(vel: f64, a: f64) -> f64 {
    let a_max = if vel > V_SWITCH {
        A_MAX * V_SWITCH / vel
    } else {
        A_MAX
    };
    let a = a.clamp(-A_MAX, a_max);
    match a {
        a if a < 0.0 && vel <= V_MIN => 0.0,
        a if a > 0.0 && vel >= V_MAX => 0.0,
        a => a,
    }
}

fn kinematic_dynamics(s: &State, a: f64, sv: f64) -> State {
    let [_, _, steer, vx, yaw, _, _] = *s;
    let beta = (LR / LWB * steer.tan()).atan();
    let (sin_b, cos_sq) = (beta.sin(), steer.cos().powi(2));
    [
        vx * (yaw + beta).cos(),                          // ẋ
        vx * (yaw + beta).sin(),                          // ẏ
        sv,                                               // δ̇
        a,                                                // v̇x
        vx / LR * sin_b,                                  // ψ̇
        a / LWB * steer.tan() + vx / (LWB * cos_sq) * sv, // ψ̈
        0.0,                                              // β̇
    ]
}

fn dynamic_dynamics(s: &State, a: f64, sv: f64) -> State {
    let [_, _, steer, vx, yaw, yr, slip] = *s;

    let vx_safe = if vx >= 0.0 {
        vx.max(V_KIN_THRESHOLD)
    } else {
        vx.min(-V_KIN_THRESHOLD)
    };

    let rear_load = G * LF + a * H;
    let front_load = G * LR - a * H;

    let yaw_acc = -MU * MASS / (vx_safe * I_Z * LWB)
        * (LF * LF * C_SF * front_load + LR * LR * C_SR * rear_load)
        * yr
        + MU * MASS / (I_Z * LWB) * (LR * C_SR * rear_load - LF * C_SF * front_load) * slip
        + MU * MASS / (I_Z * LWB) * LF * C_SF * front_load * steer;

    let slip_rate =
        (MU / (vx_safe * vx_safe * LWB) * (C_SR * rear_load * LR - C_SF * front_load * LF) - 1.0)
            * yr
            - MU / (vx_safe * LWB) * (C_SR * rear_load + C_SF * front_load) * slip
            + MU / (vx_safe * LWB) * C_SF * front_load * steer;

    [
        vx * (slip + yaw).cos(), // ẋ
        vx * (slip + yaw).sin(), // ẏ
        sv,                      // δ̇
        a,                       // v̇x
        yr,                      // ψ̇
        yaw_acc,                 // ψ̈
        slip_rate,               // β̇
    ]
}

fn dynamics(s: &State, a: f64, sv: f64) -> State {
    if s[3].abs() < V_KIN_THRESHOLD {
        kinematic_dynamics(s, a, sv)
    } else {
        dynamic_dynamics(s, a, sv)
    }
}

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&std::array::from_fn(|i| y[i] + k1[i] * dt / 2.0));
    let k3 = f(&std::array::from_fn(|i| y[i] + k2[i] * dt / 2.0));
    let k4 = f(&std::array::from_fn(|i| y[i] + k3[i] * dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn pid(target_steer: f64, target_speed: f64, vx: f64, steer: f64) -> (f64, f64) {
    let sv = if (target_steer - steer).abs() > 1e-4 {
        (target_steer - steer).signum() * STEER_VEL_MAX
    } else {
        0.0
    };
    let kp =
        if vx > 0.0 { 10.0 } else { 2.0 } * A_MAX / if target_speed > vx { V_MAX } else { -V_MIN };
    (kp * (target_speed - vx), sv)
}

impl Car {
    pub fn step(&mut self, steer: f64, speed: f64, dt: f64) {
        let (raw_a, raw_sv) = pid(steer, speed, self.velocity, self.steering);
        let a = accel_constraint(self.velocity, raw_a);
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
        let [x, y, s, vx, yaw, yr, slip] = rk4(&state, dt, |s| dynamics(s, a, sv));

        self.x = x;
        self.y = y;
        self.steering = s.clamp(STEER_MIN, STEER_MAX);
        self.velocity = vx.clamp(V_MIN, V_MAX);
        self.theta = yaw;
        self.yaw_rate = yr;
        self.slip_angle = slip;
    }
}

#[cfg(test)]
mod dynamics_tests {
    use super::*;

    /// Integrate the raw dynamics (no PID, no constraint) over `[0, t_final)`
    /// with a fixed step of `dt` using the crate's own RK4 integrator.
    #[allow(dead_code)]
    fn integrate_raw(initial: State, a: f64, sv: f64, dt: f64, t_final: f64) -> State {
        let steps = (t_final / dt).round() as usize;
        let mut s = initial;
        for _ in 0..steps {
            s = rk4(&s, dt, |st| dynamics(st, a, sv));
        }
        s
    }

    /// Integrate using constrained inputs (matches what `Car::step` does
    /// minus the PID layer) so we can feed exact accel / steer-velocity.
    fn integrate_constrained(initial: State, a: f64, sv: f64, dt: f64, t_final: f64) -> State {
        let steps = (t_final / dt).round() as usize;
        let mut s = initial;
        for _ in 0..steps {
            let ca = accel_constraint(s[3], a);
            let csv = steering_constraint(s[2], sv);
            s = rk4(&s, dt, |st| dynamics(st, ca, csv));
            s[2] = s[2].clamp(STEER_MIN, STEER_MAX);
            s[3] = s[3].clamp(V_MIN, V_MAX);
        }
        s
    }

    fn assert_state_near(got: &State, expected: &State, tol: f64, label: &str) {
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "{label}: state[{i}] mismatch: got {g:.10}, expected {e:.10} (tol {tol})"
            );
        }
    }

    // ── 1. Derivative / dynamics sanity checks ──────────────────────────

    #[test]
    fn test_kinematic_derivatives_zero_state() {
        // At zero state with zero input the kinematic model must return all zeros.
        let s: State = [0.0; 7];
        let f = kinematic_dynamics(&s, 0.0, 0.0);
        for (i, &v) in f.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "kinematic_dynamics: f[{i}] should be 0, got {v}"
            );
        }
    }

    #[test]
    fn test_dynamic_switch_threshold() {
        // Below the kinematic/dynamic switch boundary the `dynamics`
        // dispatcher must choose the kinematic model.
        let s: State = [0.0, 0.0, 0.0, V_KIN_THRESHOLD * 0.99, 0.0, 0.0, 0.0];
        let f_dispatch = dynamics(&s, 0.0, 0.0);
        let f_kin = kinematic_dynamics(&s, 0.0, 0.0);
        assert_eq!(
            f_dispatch, f_kin,
            "Should use kinematic model below threshold"
        );
    }

    #[test]
    fn test_dynamic_model_above_threshold() {
        // Above the threshold the dynamic model must be used.
        let s: State = [0.0, 0.0, 0.1, 5.0, 0.0, 0.0, 0.0];
        let f_dispatch = dynamics(&s, 1.0, 0.0);
        let f_dyn = dynamic_dynamics(&s, 1.0, 0.0);
        assert_eq!(
            f_dispatch, f_dyn,
            "Should use dynamic model above threshold"
        );
    }

    #[test]
    fn test_kinematic_straight_line() {
        // Driving straight (steer=0, slip=0) at vx=5 m/s, yaw=0:
        // ẋ = vx, ẏ = 0, rest governed by inputs.
        let s: State = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        let f = kinematic_dynamics(&s, 1.0, 0.0);
        assert!((f[0] - 5.0).abs() < 1e-10, "ẋ should equal vx");
        assert!(f[1].abs() < 1e-10, "ẏ should be 0 for straight line");
        assert!(
            (f[3] - 1.0).abs() < 1e-10,
            "v̇x should equal acceleration input"
        );
    }

    #[test]
    fn test_dynamic_derivatives_known_state() {
        // Evaluate the dynamic model at a known operating point and verify
        // key structural properties.
        let steer = 0.05;
        let vx = 8.0;
        let yaw = 0.1;
        let yr = 0.1;
        let slip = 0.02;
        let s: State = [1.0, 0.5, steer, vx, yaw, yr, slip];
        let a = 2.0;
        let sv = 0.1;
        let f = dynamic_dynamics(&s, a, sv);

        // ẋ = vx * cos(slip + yaw)
        let expected_xdot = vx * (slip + yaw).cos();
        assert!(
            (f[0] - expected_xdot).abs() < 1e-10,
            "ẋ mismatch: got {}, expected {}",
            f[0],
            expected_xdot
        );
        // ẏ = vx * sin(slip + yaw)
        let expected_ydot = vx * (slip + yaw).sin();
        assert!(
            (f[1] - expected_ydot).abs() < 1e-10,
            "ẏ mismatch: got {}, expected {}",
            f[1],
            expected_ydot
        );
        // δ̇ = sv, v̇x = a, ψ̇ = yr
        assert!((f[2] - sv).abs() < 1e-12);
        assert!((f[3] - a).abs() < 1e-12);
        assert!((f[4] - yr).abs() < 1e-12);
    }

    // ── 2. Constraint functions ─────────────────────────────────────────

    #[test]
    fn test_steering_constraint_limits() {
        // At minimum steering angle, negative velocity is clamped to 0.
        assert_eq!(steering_constraint(STEER_MIN, -1.0), 0.0);
        // At maximum steering angle, positive velocity is clamped to 0.
        assert_eq!(steering_constraint(STEER_MAX, 1.0), 0.0);
        // Within range, velocity is just clamped to [sv_min, sv_max].
        assert_eq!(steering_constraint(0.0, 100.0), STEER_VEL_MAX);
        assert_eq!(steering_constraint(0.0, -100.0), STEER_VEL_MIN);
    }

    #[test]
    fn test_accel_constraint_limits() {
        // At V_MIN, negative accel is zeroed.
        assert_eq!(accel_constraint(V_MIN, -5.0), 0.0);
        // At V_MAX, positive accel is zeroed.
        assert_eq!(accel_constraint(V_MAX, 5.0), 0.0);
        // Above V_SWITCH the positive limit drops.
        let vel = V_SWITCH * 2.0;
        let limited = accel_constraint(vel, 100.0);
        let expected = A_MAX * V_SWITCH / vel;
        assert!((limited - expected).abs() < 1e-10);
        // Negative accel clamps to -A_MAX.
        assert!((accel_constraint(5.0, -100.0) - (-A_MAX)).abs() < 1e-10);
    }

    // ── 3. Zero-init integration tests (adapted from Python suite) ──────

    const DT: f64 = 1e-4;
    const T_FINAL: f64 = 1.0;

    #[test]
    fn test_zeroinit_roll() {
        // Zero state, zero input → state must stay at zero.
        let s0: State = [0.0; 7];
        let s = integrate_constrained(s0, 0.0, 0.0, DT, T_FINAL);
        assert_state_near(&s, &s0, 1e-8, "zeroinit_roll");
    }

    #[test]
    fn test_zeroinit_decel() {
        // Zero state, strong braking input.
        // Note: V_MIN = −5.0 in rustoracer (vs −13.6 in f1tenth_gym), so
        // the velocity saturates at −5.0 before the unclamped value of
        // −6.867 would be reached.
        //
        // Phase 1 (constant accel a = −6.867):
        //   v(t) = a·t  →  hits V_MIN at t_sat = |V_MIN / a| ≈ 0.7282 s
        //   x at t_sat = ½·a·t_sat² ≈ −1.8209
        // Phase 2 (accel clamped to 0, constant v = V_MIN):
        //   remaining time = 1.0 − t_sat ≈ 0.2718 s
        //   Δx = V_MIN · remaining ≈ −1.3590
        // Total x ≈ −3.18
        let s0: State = [0.0; 7];
        let a = -0.7 * G; // ≈ −6.867 m/s²
        let sv = 0.0;
        let s = integrate_constrained(s0, a, sv, DT, T_FINAL);

        // Velocity should saturate at V_MIN.
        assert!(
            (s[3] - V_MIN).abs() < 0.05,
            "velocity should saturate at V_MIN = {V_MIN}, got {}",
            s[3]
        );
        // Car should have moved in −x.
        let t_sat = V_MIN.abs() / a.abs();
        let x_phase1 = 0.5 * a * t_sat * t_sat;
        let x_phase2 = V_MIN * (T_FINAL - t_sat);
        let expected_x = x_phase1 + x_phase2;
        assert!(
            (s[0] - expected_x).abs() < 0.1,
            "x mismatch: got {}, expected {expected_x}",
            s[0]
        );
        // Lateral state stays near zero.
        assert!(s[1].abs() < 1e-6, "y should stay ≈ 0");
        assert!(s[4].abs() < 1e-6, "yaw should stay ≈ 0");
        assert!(s[6].abs() < 1e-6, "slip should stay ≈ 0");
    }

    #[test]
    fn test_zeroinit_accel_left_steer() {
        // Zero state, accelerating + steering left.
        let s0: State = [0.0; 7];
        let a = 0.63 * G;
        let sv = 0.15;
        let s = integrate_constrained(s0, a, sv, DT, T_FINAL);

        assert!(s[0] > 2.0, "should have moved forward in x, got {}", s[0]);
        assert!(s[3] > 5.0, "should have positive velocity, got {}", s[3]);
        assert!(
            (s[2] - sv * T_FINAL).abs() < 0.01,
            "steering angle should be ≈ sv*t = {}, got {}",
            sv * T_FINAL,
            s[2]
        );
        assert!(s[4] > 0.0, "yaw should be positive, got {}", s[4]);
        assert!(s[1] > 0.0, "y should be positive, got {}", s[1]);
    }

    #[test]
    fn test_zeroinit_roll_left_steer() {
        // Zero state, zero accel, steering left.
        // No velocity → no translation; only steering angle changes.
        let s0: State = [0.0; 7];
        let a = 0.0;
        let sv = 0.15;
        let s = integrate_constrained(s0, a, sv, DT, T_FINAL);

        assert!(s[0].abs() < 1e-6, "x should stay ≈ 0, got {}", s[0]);
        assert!(s[1].abs() < 1e-6, "y should stay ≈ 0, got {}", s[1]);
        assert!(
            (s[2] - sv * T_FINAL).abs() < 1e-4,
            "steering should be ≈ 0.15, got {}",
            s[2]
        );
        assert!(s[3].abs() < 1e-6, "velocity should stay ≈ 0, got {}", s[3]);
        assert!(s[4].abs() < 1e-6, "yaw should stay ≈ 0, got {}", s[4]);
    }

    // ── 4. RK4 integrator sanity ────────────────────────────────────────

    #[test]
    fn test_rk4_constant_velocity() {
        // Constant velocity, no inputs → pure translation.
        let vx = 3.0;
        let s0: State = [0.0, 0.0, 0.0, vx, 0.0, 0.0, 0.0];
        let dt = 0.01;
        let steps = 100; // 1 second
        let mut s = s0;
        for _ in 0..steps {
            s = rk4(&s, dt, |st| dynamics(st, 0.0, 0.0));
        }
        assert!(
            (s[0] - 3.0).abs() < 0.01,
            "x after 1s should be ≈ 3.0, got {}",
            s[0]
        );
        assert!(s[1].abs() < 0.01, "y should stay ≈ 0, got {}", s[1]);
    }

    // ── 5. PID + Car::step integration tests ────────────────────────────

    #[test]
    fn test_car_step_reaches_target_speed() {
        let mut car = Car {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            velocity: 0.0,
            steering: 0.0,
            yaw_rate: 0.0,
            slip_angle: 0.0,
        };

        let target_speed = 5.0;
        let target_steer = 0.0;
        let dt = 0.01;
        for _ in 0..500 {
            car.step(target_steer, target_speed, dt);
        }

        assert!(
            (car.velocity - target_speed).abs() < 0.5,
            "velocity should converge toward {target_speed}, got {}",
            car.velocity
        );
        assert!(
            car.x > 10.0,
            "car should have moved forward significantly, got x={}",
            car.x
        );
    }

    #[test]
    fn test_car_step_steers_left() {
        let mut car = Car {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            velocity: 0.0,
            steering: 0.0,
            yaw_rate: 0.0,
            slip_angle: 0.0,
        };

        let target_speed = 5.0;
        let target_steer = 0.3;
        let dt = 0.01;
        for _ in 0..300 {
            car.step(target_steer, target_speed, dt);
        }

        assert!(car.theta > 0.0, "yaw should be positive, got {}", car.theta);
        assert!(car.y > 0.0, "y should be positive, got {}", car.y);
        assert!(
            (car.steering - target_steer).abs() < 0.05,
            "steering should converge toward {target_steer}, got {}",
            car.steering
        );
    }

    // ── 6. Performance (mirrors Python fps assertion) ───────────────────

    #[test]
    fn test_dynamics_performance() {
        use std::time::Instant;
        let s: State = [1.0, 0.5, 0.05, 8.0, 0.1, 0.1, 0.02];
        let iterations = 100_000;
        let start = Instant::now();
        let mut dummy = s;
        for _ in 0..iterations {
            dummy = dynamics(&dummy, 2.0, 0.1);
            std::hint::black_box(&dummy);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let fps = iterations as f64 / elapsed;
        assert!(
            fps > 100_000.0,
            "dynamics eval rate too low: {fps:.0} evals/s"
        );
    }
}
