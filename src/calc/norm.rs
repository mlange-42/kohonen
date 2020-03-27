use crate::data::DataFrame;

#[derive(Debug)]
pub enum Norm {
    Unity,
    Gauss,
    None,
}

#[derive(Debug)]
pub struct DeNorm {
    scale: f64,
    offset: f64,
}

pub fn normalize(
    data: &DataFrame<f64>,
    norm: &[Norm],
    scale: &[f64],
) -> (DataFrame<f64>, Vec<DeNorm>) {
    let mut counts = vec![0; data.ncols()];
    let mut params: Vec<_> = norm
        .iter()
        .map(|n| match n {
            Norm::Unity => (std::f64::MAX, std::f64::MIN),
            _ => (0.0, 0.0),
        })
        .collect();

    for row in data.iter_rows() {
        for (i, v) in row.iter().enumerate() {
            let norm = &norm[i];
            if !v.is_nan() {
                match norm {
                    Norm::Unity => {
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
    let denorm: Vec<_> = params
        .iter()
        .zip(counts)
        .zip(norm)
        .zip(scale)
        .map(|((((p1, p2), count), norm), scale)| match norm {
            Norm::Unity => DeNorm {
                scale: scale * 1.0 / (p2 - p1),
                offset: -*p1,
            },
            Norm::Gauss => {
                let sd = ((count as f64 * p2 - p1.powi(2)) / (count * (count - 1)) as f64).sqrt();
                let mean = p1 / count as f64;
                DeNorm {
                    scale: scale * 1.0 / (2.0 * sd),
                    offset: -(mean - sd),
                }
            }
            Norm::None => DeNorm {
                scale: *scale,
                offset: 0.0,
            },
        })
        .collect();

    let mut df = DataFrame::<f64>::empty(data.ncols());

    for row in data.iter_rows() {
        df.push_row_iter(
            denorm
                .iter()
                .zip(row)
                .map(|(de, v)| (v + de.offset) * de.scale),
        );
    }

    /*let denorm = denorm
    .iter()
    .map(|de| DeNorm {
        scale: 1.0 / de.scale,
        offset: -de.offset,
    })
    .collect();*/
    (df, denorm)
}

pub fn denormalize(data: &DataFrame<f64>, denorm: &[DeNorm]) -> DataFrame<f64> {
    let mut df = DataFrame::<f64>::empty(data.ncols());
    for row in data.iter_rows() {
        df.push_row_iter(
            denorm
                .iter()
                .zip(row)
                .map(|(de, v)| v / de.scale - de.offset),
        );
    }
    df
}

#[cfg(test)]
mod tests {
    use crate::calc::norm::{denormalize, normalize, DeNormalizer, Norm, Normalizer};
    use crate::data::DataFrame;
    use rand::prelude::*;
    use statistical as stats;

    #[test]
    fn normalization() {
        let mut rng = rand::thread_rng();
        let mut data = DataFrame::<f64>::empty(3);

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
            &[Norm::Unity, Norm::Gauss, Norm::None],
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
    }
}