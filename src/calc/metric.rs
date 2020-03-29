//! Distance metrics.

/// Trait for distance metrics.
pub trait Metric: Sync {
    /// Calculates the distance / dissimilarity between two vectors.
    fn distance(&self, from: &[f64], to: &[f64]) -> f64;
}

/// Squared-Euclidean distance.
pub struct SqEuclideanMetric();
impl Metric for SqEuclideanMetric {
    fn distance(&self, from: &[f64], to: &[f64]) -> f64 {
        assert_eq!(from.len(), to.len());

        let mut sum = 0.0;
        for (a, b) in from.iter().zip(to) {
            if a.is_nan() || b.is_nan() {
            } else {
                sum += (*a - *b).powi(2);
            }
        }
        sum
    }
}

/// Euclidean distance.
pub struct EuclideanMetric();
impl Metric for EuclideanMetric {
    fn distance(&self, from: &[f64], to: &[f64]) -> f64 {
        assert_eq!(from.len(), to.len());

        let mut sum = 0.0;
        for (a, b) in from.iter().zip(to) {
            if a.is_nan() || b.is_nan() {
            } else {
                sum += (*a - *b).powi(2);
            }
        }
        sum.sqrt()
    }
}

/// Tanimoto distance.
pub struct TanimotoMetric();
impl Metric for TanimotoMetric {
    fn distance(&self, from: &[f64], to: &[f64]) -> f64 {
        assert_eq!(from.len(), to.len());
        let mut counter = 0;
        let mut sum = 0.0;

        for (a, b) in from.iter().zip(to) {
            if a.is_nan() || b.is_nan() {
            } else {
                counter += 1;
                if *a >= 0.5 {
                    if *b < 0.5 {
                        sum += 1.0
                    }
                } else {
                    if *b >= 0.5 {
                        sum += 1.0
                    }
                }
            }
        }
        sum / counter as f64
    }
}

#[cfg(test)]
mod test {
    use crate::calc::metric::{EuclideanMetric, Metric, SqEuclideanMetric, TanimotoMetric};

    #[test]
    fn tanimoto() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let c = [0.0, 1.0, 1.0];
        let dist = TanimotoMetric().distance(&a, &b);
        assert_eq!(dist, 1.0);
        let dist = TanimotoMetric().distance(&a, &c);
        assert_eq!(dist, 2.0 / 3.0);
    }
    #[test]
    fn distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 2.0, 2.0];
        let dist = SqEuclideanMetric().distance(&a, &b);
        assert_eq!(dist, 12.0);
        let dist = EuclideanMetric().distance(&a, &b);
        assert_eq!(dist, 12f64.sqrt());
    }
}
