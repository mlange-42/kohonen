//! Distance metrics.
/*
/// Trait for distance metrics.
pub trait Metric: Sync {
    /// Calculates the distance / dissimilarity between two vectors.
    fn distance(&self, from: &[f64], to: &[f64]) -> f64;
}*/

use crate::ParseEnumError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metric {
    SqEuclidean,
    Euclidean,
    Tanimoto,
}

impl Metric {
    pub fn distance(&self, from: &[f64], to: &[f64]) -> f64 {
        assert_eq!(from.len(), to.len());
        match self {
            Metric::SqEuclidean => {
                let mut sum = 0.0;
                for (a, b) in from.iter().zip(to) {
                    if a.is_nan() || b.is_nan() {
                    } else {
                        sum += (*a - *b).powi(2);
                    }
                }
                sum
            }
            Metric::Euclidean => {
                let mut sum = 0.0;
                for (a, b) in from.iter().zip(to) {
                    if a.is_nan() || b.is_nan() {
                    } else {
                        sum += (*a - *b).powi(2);
                    }
                }
                sum.sqrt()
            }
            Metric::Tanimoto => {
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
                        } else if *b >= 0.5 {
                            sum += 1.0
                        }
                    }
                }
                sum / counter as f64
            }
        }
    }
}
impl FromStr for Metric {
    type Err = ParseEnumError;
    /// Parse a string to a `Metric`.
    ///
    /// Accepts `"euclidean" | "tanimoto"`.
    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "euclidean" => Ok(Metric::Euclidean),
            "tanimoto" => Ok(Metric::Tanimoto),
            _ => Err(ParseEnumError(format!(
                "Not a metric: {}. Must be one of (euclidean|tanimoto)",
                str
            ))),
        }
    }
}

/*
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
*/

#[cfg(test)]
mod test {
    use crate::calc::metric::Metric;

    #[test]
    fn tanimoto() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let c = [0.0, 1.0, 1.0];
        let dist = Metric::Tanimoto.distance(&a, &b);
        assert!((dist - 1.0).abs() < std::f64::EPSILON);
        let dist = Metric::Tanimoto.distance(&a, &c);
        assert!((dist - 2.0 / 3.0).abs() < std::f64::EPSILON);
    }
    #[test]
    fn distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 2.0, 2.0];
        let dist = Metric::SqEuclidean.distance(&a, &b);
        assert!((dist - 12.0).abs() < std::f64::EPSILON);
        let dist = Metric::Euclidean.distance(&a, &b);
        assert!((dist - 12f64.sqrt()).abs() < std::f64::EPSILON);
    }
}
