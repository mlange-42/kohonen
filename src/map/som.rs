//! Super-SOM for flexible use as unsupervised or supervised SOM. Core types.

use crate::calc::metric::{Metric, SqEuclideanMetric};
use crate::calc::neighborhood::Neighborhood;
use crate::calc::nn;
use crate::data::DataFrame;
use crate::ParseEnumError;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp;

/// SOM training parameters
#[derive(Serialize, Deserialize)]
pub struct SomParams {
    epochs: u32,
    //metric: M,
    neighborhood: Neighborhood,
    alpha: DecayParam,
    radius: DecayParam,
    decay: DecayParam,
    layers: Vec<Layer>,
    start_columns: Vec<usize>,
}

impl SomParams {
    /// Creates parameters for a simple SOM with a simple layer.
    pub fn simple(
        epochs: u32,
        neighborhood: Neighborhood,
        alpha: DecayParam,
        radius: DecayParam,
        decay: DecayParam,
    ) -> Self {
        SomParams {
            epochs,
            neighborhood,
            alpha,
            radius,
            decay,
            layers: vec![],
            start_columns: vec![0],
        }
    }

    /// Creates parameters for a multi-layers SOM (Super-SOM) using the X-Y-Fused algorithm (XYF).
    pub fn xyf(
        epochs: u32,
        neighborhood: Neighborhood,
        alpha: DecayParam,
        radius: DecayParam,
        decay: DecayParam,
        layers: Vec<Layer>,
    ) -> Self {
        let start_cols = Self::calc_start_columns(&layers);
        SomParams {
            epochs,
            neighborhood,
            alpha,
            radius,
            decay,
            layers,
            start_columns: start_cols,
        }
    }

    /// Returns a reference to the layer definitions
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    pub fn start_columns(&self) -> &[usize] {
        &self.start_columns
    }

    fn calc_start_columns(layers: &[Layer]) -> Vec<usize> {
        let mut result = vec![0; layers.len()];
        let mut start_col = 0;
        for (i, lay) in layers.iter().enumerate() {
            result[i] = start_col;
            start_col += lay.ncols();
        }
        result
    }
}

/// Layer definition for multi-layered SOMs.
#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    ncols: usize,
    weight: f64,
    categorical: bool,
}
impl Layer {
    /// Creates a new layer.
    pub fn new(ncols: usize, weight: f64, categorical: bool) -> Self {
        Layer {
            ncols,
            weight,
            categorical,
        }
    }
    /// Creates a new continuous layer.
    pub fn cont(ncols: usize, weight: f64) -> Self {
        Self::new(ncols, weight, false)
    }
    /// Creates a new categorical layer.
    pub fn cat(ncols: usize, weight: f64) -> Self {
        Self::new(ncols, weight, true)
    }
    /// The number of data columns of the layer.
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// The weight of the layer.
    pub fn weight(&self) -> f64 {
        self.weight
    }
    /// If the layer is categorical.
    pub fn categorical(&self) -> bool {
        self.categorical
    }
}

/// Decay functions for learing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Linear decay
    Linear,
    /// Exponential decay
    Exponential,
}
impl DecayFunction {
    pub fn from_string(str: &str) -> Result<DecayFunction, ParseEnumError> {
        match str {
            "lin" => Ok(DecayFunction::Linear),
            "exp" => Ok(DecayFunction::Exponential),
            _ => Err(ParseEnumError(format!(
                "Not a decay function: {}. Must be one of (lin|exp)",
                str
            ))),
        }
    }
}
/// Decay parameters for learing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayParam {
    start: f64,
    end: f64,
    function: DecayFunction,
}
impl DecayParam {
    /// Creates a learning parameter from start and end value and decay function.
    pub fn new(start: f64, end: f64, function: DecayFunction) -> Self {
        DecayParam {
            start,
            end,
            function,
        }
    }
    /// Creates a linearly decaying learning parameter from start and end value.
    pub fn lin(start: f64, end: f64) -> Self {
        DecayParam {
            start,
            end,
            function: DecayFunction::Linear,
        }
    }
    /// Creates a exponentially decaying learning parameter from start and end value.
    pub fn exp(start: f64, end: f64) -> Self {
        DecayParam {
            start,
            end,
            function: DecayFunction::Exponential,
        }
    }
    /// Get the parameter's value for the given training episode.
    pub fn get(&self, epoch: u32, max_epochs: u32) -> f64 {
        match self.function {
            DecayFunction::Linear => {
                let frac = epoch as f64 / max_epochs as f64;
                self.start + frac * (self.end - self.start)
            }
            DecayFunction::Exponential => {
                let rate = (1.0 / -(max_epochs as f64)) * (self.end / self.start).ln();
                self.start * (-rate * epoch as f64).exp()
            }
        }
    }
}

/// Super-SOM core type.
#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Som {
    dims: usize,
    nrows: usize,
    ncols: usize,
    weights: DataFrame,
    params: SomParams,
    epoch: u32,
    #[serde(skip_serializing)]
    distances_sq: DataFrame,
}

#[allow(dead_code)]
impl Som {
    /// Creates a new SOM or Super-SOM
    pub fn new(names: &[&str], nrows: usize, ncols: usize, params: SomParams) -> Self {
        let mut som = Som {
            dims: names.len(),
            nrows,
            ncols,
            weights: DataFrame::filled(nrows * ncols, names, 0.0),
            distances_sq: Self::calc_distance_matix(nrows, ncols),
            params,
            epoch: 0,
        };
        som.init_weights();
        som
    }

    /// Returns a reference to the SOM's parameters.
    pub fn params(&self) -> &SomParams {
        &self.params
    }

    /// Initialize weights. Called by the constructor automatically (may change!).
    pub fn init_weights(&mut self) {
        let mut rng = rand::thread_rng();
        let cols = self.weights.ncols();
        for row in self.weights.iter_rows_mut() {
            for c in 0..cols {
                row[c] = rng.gen_range(0.0, 1.0);
            }
        }
    }

    /// Pre-calculates the unit-to-unit distance matrix.
    fn calc_distance_matix(nrows: usize, ncols: usize) -> DataFrame {
        let metric = SqEuclideanMetric();
        let mut df = DataFrame::filled(nrows * ncols, &vec![""; nrows * ncols], 0.0);
        for r1 in 0..nrows {
            for c1 in 0..ncols {
                let idx1 = r1 * ncols + c1;
                for r2 in 0..nrows {
                    for c2 in 0..ncols {
                        let idx2 = r2 * ncols + c2;
                        df.set(
                            idx1,
                            idx2,
                            metric.distance(&[r1 as f64, c1 as f64], &[r2 as f64, c2 as f64]),
                        );
                    }
                }
            }
        }
        df
    }
    /// Returns (row, col) for a given raw data index.
    pub fn to_row_col(&self, index: usize) -> (usize, usize) {
        (index / self.ncols, index % self.ncols)
    }
    /// Returns the raw data index for (row, col).
    pub fn to_index(&self, row: i32, col: i32) -> usize {
        (row * self.ncols as i32 + col) as usize
    }
    /// Returns a reference to the units weights data frame.
    pub fn weights(&self) -> &DataFrame {
        &self.weights
    }
    /// Returns a reference to the weights of the unit at (row, col).
    pub fn weights_at(&self, row: usize, col: usize) -> &[f64] {
        self.weights.get_row(self.to_index(row as i32, col as i32))
    }
    /// The number of columns (width) of the SOM.
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// The number of rows (height) of the SOM.
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// The size og the SOM as (rows, cols).
    pub fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Trains the SOM for one epoch. Updates learning parameters
    pub fn epoch(&mut self, samples: &DataFrame, count: Option<usize>) -> Option<()> {
        if self.epoch >= self.params.epochs {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<_> = (0..samples.nrows()).collect();
        rng.shuffle(&mut indices);

        let cnt = cmp::min(count.unwrap_or_else(|| samples.nrows()), samples.nrows());

        for idx in indices.iter().take(cnt) {
            let sample = samples.get_row(*idx);
            self.train(sample);
        }

        self.decay_weights();

        self.epoch += 1;

        Some(())
    }

    /// Decays unit weights.
    fn decay_weights(&mut self) {
        let means = self.weights.means();
        let cols = self.weights.ncols();
        let decay = self.params.decay.get(self.epoch, self.params.epochs);
        for row in self.weights.iter_rows_mut() {
            for c in 0..cols {
                let v = row[c];
                let m = means[c];
                row[c] = v - decay * (v - m);
            }
        }
    }

    /// Trains the SOM for a single sample.
    fn train(&mut self, sample: &[f64]) {
        let params = &self.params;
        let (nearest, _) = if params.layers.len() == 0 {
            nn::nearest_neighbor(sample, &self.weights)
        } else if params.layers.len() == 1 {
            if params.layers[0].categorical {
                nn::nearest_neighbor_tanimoto(sample, &self.weights)
            } else {
                nn::nearest_neighbor(sample, &self.weights)
            }
        } else {
            nn::nearest_neighbor_xyf(sample, &self.weights, &params.layers)
        };
        let (row, col) = self.to_row_col(nearest);

        let alpha = self.params.alpha.get(self.epoch, self.params.epochs);
        let radius = self.params.radius.get(self.epoch, self.params.epochs);
        let neigh = &self.params.neighborhood;
        let radius_inf_sq = (1.0 / radius).powi(2);
        let search_rad = radius * neigh.radius();
        let search_rad_i = search_rad.floor() as i32;
        let search_rad_sq = search_rad.powi(2);

        let r_min = cmp::max(0, row as i32 - search_rad_i);
        let r_max = cmp::min(self.nrows as i32 - 1, row as i32 + search_rad_i);
        let c_min = cmp::max(0, col as i32 - search_rad_i);
        let c_max = cmp::min(self.ncols as i32 - 1, col as i32 + search_rad_i);

        for r in r_min..=r_max {
            for c in c_min..=c_max {
                let index = self.to_index(r, c);
                let dist_sq = *self.distances_sq.get(nearest, index) as f64;
                if dist_sq <= search_rad_sq {
                    let weight = neigh.weight(radius_inf_sq * dist_sq);
                    for i in 0..self.dims {
                        let smp = sample[i];
                        if !smp.is_nan() {
                            let value = *self.weights.get(index, i);
                            self.weights
                                .set(index, i, value + weight * alpha * (smp - value));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::Neighborhood;
    use crate::data::DataFrame;
    use crate::map::som::{DecayParam, Som, SomParams};
    use rand::Rng;

    #[test]
    fn create_som() {
        let params = SomParams::simple(
            100,
            Neighborhood::Gauss,
            DecayParam::lin(0.2, 0.01),
            DecayParam::lin(1.0, 0.5),
            DecayParam::lin(0.2, 0.001),
        );
        let som = Som::new(&["A", "B", "C"], 3, 3, params);
        assert_eq!(som.distances_sq.get(0, 8), &8.0);
    }

    #[test]
    fn train_step() {
        let params = SomParams::simple(
            100,
            Neighborhood::Gauss,
            DecayParam::lin(0.2, 0.01),
            DecayParam::lin(1.0, 0.5),
            DecayParam::lin(0.2, 0.001),
        );
        let mut som = Som::new(&["A", "B", "C"], 4, 4, params);

        som.train(&[1.0, 1.0, 1.0]);
    }
    #[test]
    fn train_epoch() {
        let cols = ["A", "B", "C", "D", "E"];
        let params = SomParams::simple(
            10,
            Neighborhood::Gauss,
            DecayParam::lin(0.2, 0.01),
            DecayParam::lin(5.0, 0.5),
            DecayParam::exp(0.2, 0.001),
        );
        let mut som = Som::new(&cols, 16, 16, params);

        let mut rng = rand::thread_rng();
        let mut data = DataFrame::empty(&cols);

        for _i in 0..100 {
            data.push_row(&[
                rng.gen_range(0.7, 0.8),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            ]);
        }

        while let Some(()) = som.epoch(&data, None) {}

        /*for row in som.weights.iter_rows() {
            println!("{:?}", row);
        }*/
    }

    #[test]
    fn linear_decay() {
        let decay = DecayParam::lin(1.0, 0.1);

        assert!((decay.get(0, 100) - 1.0).abs() < 0.0001);
        assert!((decay.get(100, 100) - 0.1).abs() < 0.0001);
    }
    #[test]
    fn exponential_decay() {
        let decay = DecayParam::exp(1.0, 0.01);

        assert!((decay.get(0, 100) - 1.0).abs() < 0.0001);
        assert!((decay.get(100, 100) - 0.01).abs() < 0.0001);
    }
}
