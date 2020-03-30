//! Normalization and de-normalization of data.

use crate::data::DataFrame;
use crate::ParseEnumError;
use serde::{Deserialize, Serialize};

/// Normalization types.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum Norm {
    /// Normalize to [0, 1].
    Unit,
    /// Normalize to a mean of 0.5 and standard deviation of 0.5.
    Gauss,
    /// No normalization
    None,
}

impl Norm {
    pub fn from_string(str: &str) -> Result<Norm, ParseEnumError> {
        match str {
            "unit" => Ok(Norm::Unit),
            "gauss" => Ok(Norm::Gauss),
            "none" => Ok(Norm::None),
            _ => Err(ParseEnumError(format!(
                "Not a normalizer: {}. Must be one of (unit|gauss|none)",
                str
            ))),
        }
    }
}

/// De-normalization parameters. Obtained from [`normalize`](fn.normalize.html).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTransform {
    scale: f64,
    offset: f64,
}

impl LinearTransform {
    pub fn transform(&self, value: f64) -> f64 {
        value * self.scale + self.offset
    }
    pub fn inverse(&self) -> LinearTransform {
        LinearTransform {
            scale: 1.0 / self.scale,
            offset: -self.offset / self.scale,
        }
    }
}

/// Normalize a data frame, with a [`Norm`](struct.Norm.html) and scale per column.
/// # Returns
/// A tuple of: (normalized data frame, vector of [`LinearTransform`](struct.LinearTransform.html) for de-normalization, one per column).
pub fn normalize(
    data: &DataFrame,
    norm: &[Norm],
    scale: &[f64],
) -> (DataFrame, Vec<LinearTransform>) {
    let mut counts = vec![0; data.ncols()];
    let mut params: Vec<_> = norm
        .iter()
        .map(|n| match n {
            Norm::Unit => (std::f64::MAX, std::f64::MIN),
            _ => (0.0, 0.0),
        })
        .collect();

    for row in data.iter_rows() {
        for (i, v) in row.iter().enumerate() {
            if !v.is_nan() {
                let norm = &norm[i];
                match norm {
                    Norm::Unit => {
                        if *v < params[i].0 {
                            params[i].0 = *v
                        }
                        if *v > params[i].1 {
                            params[i].1 = *v
                        }
                    }
                    Norm::Gauss => {
                        params[i].0 += *v;
                        params[i].1 += v.powi(2);
                    }
                    Norm::None => {}
                }
                counts[i] += 1;
            }
        }
    }
    //println!("Params: {:?}", params);
    //println!("Counts: {:?}", counts);
    let denorm: Vec<_> = params
        .iter()
        .zip(counts)
        .zip(norm)
        .zip(scale)
        .map(|((((p1, p2), count), norm), scale)| match norm {
            Norm::Unit => {
                let sc = scale / (p2 - p1);
                LinearTransform {
                    //scale: scale * 1.0 / (p2 - p1),
                    //offset: -*p1,
                    scale: sc,
                    offset: -*p1 * sc,
                }
            }
            Norm::Gauss => {
                let sd = ((count as f64 * p2 - p1.powi(2)) / (count * (count - 1)) as f64).sqrt();
                let mean = p1 / count as f64;
                let sc = scale / (2.0 * sd);
                LinearTransform {
                    //scale: scale * 1.0 / (2.0 * sd),
                    //offset: -(mean - sd),
                    scale: sc,
                    offset: -(mean - sd) * sc,
                }
            }
            Norm::None => LinearTransform {
                scale: *scale,
                offset: 0.0,
            },
        })
        .collect();

    let cols: Vec<_> = data.names().iter().map(|x| &**x).collect();
    let mut df = DataFrame::empty(&cols);

    for row in data.iter_rows() {
        df.push_row_iter(
            denorm
                .iter()
                .zip(row)
                //.map(|(de, v)| (v + de.offset) * de.scale),
                .map(|(de, v)| de.transform(*v)),
        );
    }

    let denorm = denorm.iter().map(|de| de.inverse()).collect();
    (df, denorm)
}

/// De-normalize a data frame, with a [`LinearTransform`](struct.LinearTransform.html) per column, as obtained from [`normalize`](fn.normalize.html).
/// # Returns
/// A de-normalized data frame
pub fn denormalize(data: &DataFrame, denorm: &[LinearTransform]) -> DataFrame {
    assert_eq!(data.ncols(), denorm.len());
    let cols: Vec<_> = data.names().iter().map(|x| &**x).collect();
    let mut df = DataFrame::empty(&cols);
    for row in data.iter_rows() {
        df.push_row_iter(
            denorm
                .iter()
                .zip(row)
                //.map(|(de, v)| v / de.scale - de.offset),
                .map(|(de, v)| de.transform(*v)),
        );
    }
    df
}

/// De-normalize columns of a data frame, with a [`LinearTransform`](struct.LinearTransform.html) per column, as obtained from [`normalize`](fn.normalize.html).
/// # Returns
/// A de-normalized data frame
pub fn denormalize_columns(
    data: &DataFrame,
    columns: &[usize],
    denorm: &[LinearTransform],
) -> DataFrame {
    assert_eq!(columns.len(), denorm.len());
    let cols: Vec<_> = columns.iter().map(|i| &data.names()[*i][..]).collect();
    let mut df = DataFrame::empty(&cols);
    for row in data.iter_rows() {
        df.push_row_iter(
            columns
                .iter()
                .zip(denorm)
                .map(|(col, de)| de.transform(row[*col])),
        );
    }
    df
}

#[cfg(test)]
mod tests {
    use crate::calc::norm::{denormalize, denormalize_columns, normalize, Norm};
    use crate::data::DataFrame;
    use rand::prelude::*;
    use statistical as stats;

    #[test]
    fn normalization() {
        let mut rng = rand::thread_rng();
        let mut data = DataFrame::empty(&["A", "B", "C"]);

        let norm = rand::distributions::Normal::new(1.0, 2.0);
        for _i in 0..20 {
            data.push_row(&[
                rng.gen_range(-1.0, 5.0),
                norm.sample(&mut rng),
                rng.gen_range(-1.0, 1.0),
            ]);
        }

        let (df, denorm) = normalize(
            &data,
            &[Norm::Unit, Norm::Gauss, Norm::None],
            &[1.0, 1.0, 0.5],
        );

        assert_eq!(data.nrows(), df.nrows());
        assert_eq!(data.ncols(), df.ncols());

        let ranges = df.ranges();

        assert!(ranges[0].0 > -0.0001 && ranges[0].0 < 0.0001);
        assert!(ranges[0].1 > 0.9999 && ranges[0].1 < 1.0001);
        let mean_1 = stats::mean(&df.copy_column(1));
        let sd_1 = stats::standard_deviation(&df.copy_column(1), None);
        assert!(mean_1 > 0.4999 && mean_1 < 0.5001);
        assert!(sd_1 > 0.4999 && sd_1 < 0.5001);

        let df2 = denormalize(&df, &denorm);

        for (row1, row2) in data.iter_rows().zip(df2.iter_rows()) {
            for (v1, v2) in row1.iter().zip(row2) {
                assert!((v1 - v2).abs() < 0.00001);
            }
        }

        let df3 = denormalize_columns(&df, &[0, 1], &denorm[0..2]);

        assert_eq!(df3.ncols(), 2);
        for (row1, row2) in data.iter_rows().zip(df3.iter_rows()) {
            for (v1, v2) in row1.iter().zip(row2) {
                assert!((v1 - v2).abs() < 0.00001);
            }
        }
    }
}
