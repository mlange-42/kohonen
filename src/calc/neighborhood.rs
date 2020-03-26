pub trait Neighborhood {
    fn weight(&self, distance_sq: f64) -> f64;
    fn radius(&self) -> f64;
}

pub struct GaussNeighborhood();

impl Neighborhood for GaussNeighborhood {
    fn weight(&self, distance_sq: f64) -> f64 {
        if distance_sq == 0.0 {
            1.0
        } else {
            (-0.5 * distance_sq).exp()
        }
    }
    fn radius(&self) -> f64 {
        3.0
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::{GaussNeighborhood, Neighborhood};

    #[test]
    fn gauss() {
        let neigh = GaussNeighborhood();
        assert_eq!(neigh.weight(0.0), 1.0);
        assert!(neigh.weight(3.0 * 3.0) < 0.12);
    }

    #[test]
    fn distance_scaling() {
        let dist = 2_f32;
        let scale = 2_f32;

        let dist_sq = dist.powi(2);
        let dist_sq_sc = (1.0 / scale).powi(2) * (dist * scale).powi(2);
        let dist_sq_sc_2 = ((1.0 / scale) * (dist * scale)).powi(2);

        assert_eq!(dist_sq, dist_sq_sc);
        assert_eq!(dist_sq_sc, dist_sq_sc_2);
    }
}
