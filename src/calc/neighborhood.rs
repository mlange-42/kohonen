//! Neighborhoods (i.e. kernels), for effect on nearby SOM-units.

use crate::ParseEnumError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Neighborhoods: 4 or 8 neighbors.
#[derive(Debug, Clone)]
pub enum Neighbors {
    Neighbors4,
    Neighbors8,
}
impl FromStr for Neighbors {
    type Err = ParseEnumError;

    /// Parse a string to a `Neighbors`.
    ///
    /// Accepts `"4" | "n4" | "N4" | "Neighbors4" | "8" | "n8" | "N8" | "Neighbors8"`.
    fn from_str(str: &str) -> Result<Neighbors, ParseEnumError> {
        match str {
            "4" | "n4" | "N4" | "Neighbors4" => Ok(Neighbors::Neighbors4),
            "8" | "n8" | "N8" | "Neighbors8" => Ok(Neighbors::Neighbors8),
            _ => Err(ParseEnumError(format!(
                "Not a Neighbors: {}. Must be one of (n4|n8)",
                str
            ))),
        }
    }
}

/// Neighborhood functions / kernels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Neighborhood {
    Gauss,
    Triangular,
    Epanechnikov,
    Quartic,
    Triweight,
}
impl Neighborhood {
    /// Calculates the weight, depending on the distance.
    pub fn weight(&self, distance: f64) -> f64 {
        match self {
            Neighborhood::Gauss => {
                if distance == 0.0 {
                    1.0
                } else {
                    (-0.5 * distance * distance).exp()
                }
            }
            Neighborhood::Triangular => {
                if distance >= 1.0 {
                    0.0
                } else {
                    1.0 - distance
                }
            }
            Neighborhood::Epanechnikov => {
                if distance >= 1.0 {
                    0.0
                } else {
                    1.0 - distance * distance
                }
            }
            Neighborhood::Quartic => {
                if distance >= 1.0 {
                    0.0
                } else {
                    (1.0 - distance * distance).powi(2)
                }
            }
            Neighborhood::Triweight => {
                if distance >= 1.0 {
                    0.0
                } else {
                    (1.0 - distance * distance).powi(3)
                }
            }
        }
    }
    /// Maximum search distance in the SOM.
    pub fn radius(&self) -> f64 {
        match self {
            Neighborhood::Gauss => 3.0,
            _ => 1.0,
        }
    }
}
impl FromStr for Neighborhood {
    type Err = ParseEnumError;

    /// Parse a string to a `Neighborhood`.
    ///
    /// Accepts `gauss | triangular | epanechnikov | quartic | triweight`.
    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "gauss" => Ok(Neighborhood::Gauss),
            "triangular" => Ok(Neighborhood::Triangular),
            "epanechnikov" => Ok(Neighborhood::Epanechnikov),
            "quartic" => Ok(Neighborhood::Quartic),
            "triweight" => Ok(Neighborhood::Triweight),
            _ => Err(ParseEnumError(format!(
                "Not a neighborhood: {}. Must be one of (gauss|<todo>)",
                str
            ))),
        }
    }
}
/*
impl EnumFromString for Neighborhood {
    /// Parse a string to a `Neighborhood`.
    ///
    /// Accepts `"gauss" | <TODO>`.
    fn from_string(str: &str) -> Result<Neighborhood, ParseEnumError> {
        match str {
            "gauss" => Ok(Neighborhood::Gauss),
            "triangular" => Ok(Neighborhood::Triangular),
            "epanechnikov" => Ok(Neighborhood::Epanechnikov),
            "quartic" => Ok(Neighborhood::Quartic),
            "triweight" => Ok(Neighborhood::Triweight),
            _ => Err(ParseEnumError(format!(
                "Not a neighborhood: {}. Must be one of (gauss|<todo>)",
                str
            ))),
        }
    }
}*/

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::Neighborhood;

    #[test]
    fn gauss() {
        let neigh = Neighborhood::Gauss;
        assert!((neigh.weight(0.0) - 1.0).abs() < std::f64::EPSILON);
        assert!(neigh.weight(3.0 * 3.0) < 0.12);
    }

    #[test]
    fn distance_scaling() {
        let dist = 2_f32;
        let scale = 2_f32;

        let dist_sq = dist.powi(2);
        let dist_sq_sc = (1.0 / scale).powi(2) * (dist * scale).powi(2);
        let dist_sq_sc_2 = ((1.0 / scale) * (dist * scale)).powi(2);

        assert!((dist_sq - dist_sq_sc).abs() < std::f32::EPSILON);
        assert!((dist_sq_sc - dist_sq_sc_2).abs() < std::f32::EPSILON);
    }
}
