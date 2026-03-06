use numpy::{IntoPyArray, PyArray1, PyArray3, ToPyArray};
use pyo3::prelude::*;

use crate::Sim;

#[pymodule]
mod rustoracer {
    use numpy::PyReadonlyArray1;

    use super::*;

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
            let o = self.sim.step(&actions.as_slice().unwrap());
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
            let raw = xy.as_slice().unwrap();
            let h = self.sim.map.img.height();
            let inv_res = 1.0 / self.sim.map.res;
            let ox = self.sim.map.ox;
            let oy = self.sim.map.oy;
            let out: Vec<f64> = raw
                .chunks(2)
                .flat_map(|p| {
                    [
                        (p[0] - ox) * inv_res,
                        (h - 1) as f64 - (p[1] - oy) * inv_res,
                    ]
                })
                .collect();
            out.into_pyarray(py)
        }

        fn car_pixels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let out: Vec<f64> = self
                .sim
                .cars
                .iter()
                .flat_map(|car| {
                    self.sim
                        .map
                        .car_pixels(car)
                        .flat_map(|(px, py)| [px as f64, py as f64])
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
                .flat_map(|p| p.iter().copied())
                .collect();
            flat.into_pyarray(py)
        }
    }
}
