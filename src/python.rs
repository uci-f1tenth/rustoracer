use crate::Sim;
use numpy::{IntoPyArray, PyArray1, PyArray3, ToPyArray};
use pyo3::prelude::*;

#[pymodule]
mod rustoracer {
    use super::*;
    use numpy::PyReadonlyArray1;

    #[pyclass]
    struct PySim {
        sim: Sim,
    }

    #[pymethods]
    impl PySim {
        #[new]
        fn new(yaml: &str, n: usize, max_steps: u32) -> Self {
            Self {
                sim: Sim::new(yaml, n, max_steps),
            }
        }

        fn seed(&mut self, seed: u64) {
            self.sim.seed(seed);
        }

        fn reset<'py>(
            &mut self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let o = self.sim.reset();
            (
                o.scans.to_pyarray(py),
                o.rewards.to_pyarray(py),
                o.terminated.to_pyarray(py),
                o.truncated.to_pyarray(py),
                o.state.to_pyarray(py),
            )
        }

        fn easy_step<'py>(
            &mut self,
            py: Python<'py>,
            actions: PyReadonlyArray1<f64>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let transformed: Vec<f64> = actions
                .as_slice()
                .unwrap()
                .chunks(2)
                .enumerate()
                .flat_map(|(i, c)| {
                    let ds = if c[0] > self.sim.cars[i].steering {
                        1.0
                    } else {
                        -1.0
                    };
                    let dv = if c[1] > self.sim.cars[i].velocity {
                        1.0
                    } else {
                        -1.0
                    };
                    [ds, dv]
                })
                .collect();
            let o = self.sim.step(&transformed);
            (
                o.scans.to_pyarray(py),
                o.rewards.to_pyarray(py),
                o.terminated.to_pyarray(py),
                o.truncated.to_pyarray(py),
                o.state.to_pyarray(py),
            )
        }

        fn step<'py>(
            &mut self,
            py: Python<'py>,
            actions: PyReadonlyArray1<f64>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let o = self.sim.step(actions.as_slice().unwrap());
            (
                o.scans.to_pyarray(py),
                o.rewards.to_pyarray(py),
                o.terminated.to_pyarray(py),
                o.truncated.to_pyarray(py),
                o.state.to_pyarray(py),
            )
        }

        fn render<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
            let (buf, h, w) = crate::render::render_rgb(&self.sim.map, &self.sim.cars);
            numpy::ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), buf)
                .unwrap()
                .into_pyarray(py)
        }

        fn observe<'py>(
            &mut self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let o = self.sim.observe();
            (
                o.scans.to_pyarray(py),
                o.rewards.to_pyarray(py),
                o.terminated.to_pyarray(py),
                o.truncated.to_pyarray(py),
                o.state.to_pyarray(py),
            )
        }

        #[getter]
        fn obs_dim(&self) -> usize {
            self.sim.obs_dim
        }
        #[getter]
        fn n_beams(&self) -> usize {
            self.sim.n_beams
        }
        #[getter]
        fn min_range(&self) -> f64 {
            self.sim.min_range
        }
        #[getter]
        fn max_range(&self) -> f64 {
            self.sim.max_range
        }
        #[getter]
        fn fov(&self) -> f64 {
            self.sim.fov
        }

        fn world_to_pixels<'py>(
            &self,
            py: Python<'py>,
            xy: PyReadonlyArray1<f64>,
        ) -> Bound<'py, PyArray1<f64>> {
            let (h, inv, ox, oy) = (
                self.sim.map.img.height(),
                1.0 / self.sim.map.res,
                self.sim.map.ox,
                self.sim.map.oy,
            );
            let out: Vec<f64> = xy
                .as_slice()
                .unwrap()
                .chunks(2)
                .flat_map(|p| [(p[0] - ox) * inv, (h - 1) as f64 - (p[1] - oy) * inv])
                .collect();
            out.into_pyarray(py)
        }

        fn car_pixels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let out: Vec<f64> = self
                .sim
                .cars
                .iter()
                .flat_map(|c| {
                    self.sim
                        .map
                        .car_pixels(c)
                        .flat_map(|(x, y)| [x as f64, y as f64])
                })
                .collect();
            out.into_pyarray(py)
        }

        #[getter]
        fn skeleton<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let flat: Vec<f64> = self
                .sim
                .map
                .skeleton
                .points
                .iter()
                .flat_map(|p| *p)
                .collect();
            flat.into_pyarray(py)
        }
    }
}
