//! Nearest-neighbor search.

use crate::data::DataFrame;

use crate::calc::metric::{EuclideanMetric, Metric, SqEuclideanMetric, TanimotoMetric};
use crate::map::som::Layer;

#[allow(dead_code)]
const EUCLIDEAN: EuclideanMetric = EuclideanMetric();
#[allow(dead_code)]
const EUCLIDEAN_SQ: SqEuclideanMetric = SqEuclideanMetric();
#[allow(dead_code)]
const TANIMOTO: TanimotoMetric = TanimotoMetric();

/// Nearest-neighbor by Euclidean distance.
/// Dimensions with NA values are ignored.
/// # Returns
/// (index, distance)
pub fn nearest_neighbor(from: &[f64], to: &DataFrame) -> (usize, f64) {
    assert_eq!(from.len(), to.ncols());

    let mut min_dist = std::f64::MAX;
    let mut min_idx: usize = 0;
    for (idx_to, row_to) in to.iter_rows().enumerate() {
        let dist = EUCLIDEAN_SQ.distance(from, row_to);
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx_to
        }
    }
    (min_idx, min_dist.sqrt())
}

/// Nearest-neighbor by Tanimoto distance.
/// Dimensions with NA values are ignored.
/// # Returns
/// (index, distance)
pub fn nearest_neighbor_tanimoto(from: &[f64], to: &DataFrame) -> (usize, f64) {
    assert_eq!(from.len(), to.ncols());

    let mut min_dist = std::f64::MAX;
    let mut min_idx: usize = 0;
    for (idx_to, row_to) in to.iter_rows().enumerate() {
        let dist = TANIMOTO.distance(from, row_to);
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx_to
        }
    }
    (min_idx, min_dist.sqrt())
}

/// Nearest-neighbor for XYF-maps. Layers determine distance metrics and weighting.
/// Dimensions with NA values are ignored.
/// # Returns
/// (index, weighted-distance)
pub fn nearest_neighbor_xyf(from: &[f64], to: &DataFrame, layers: &[Layer]) -> (usize, f64) {
    assert_eq!(from.len(), to.ncols());

    let mut min_dist = std::f64::MAX;
    let mut min_idx: usize = 0;
    for (idx_to, row_to) in to.iter_rows().enumerate() {
        let mut start = 0_usize;
        let mut dist = 0.0;
        for layer in layers {
            let end = start + layer.ncols();
            let d = if layer.categorical() {
                TANIMOTO.distance(&from[start..end], &row_to[start..end])
            } else {
                EUCLIDEAN.distance(&from[start..end], &row_to[start..end])
            };
            if !d.is_nan() {
                dist += d * layer.weight();
            }

            start = end;
        }
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx_to
        }
    }
    (min_idx, min_dist)
}

/// Nearest-neighbors for multiple starting points, by Euclidean distance.
/// # Returns
/// Vec(index, weighted-distance)
pub fn nearest_neighbors(
    from: &DataFrame,
    to: &DataFrame,
    mut result: Vec<(usize, f64)>,
) -> Vec<(usize, f64)> {
    assert_eq!(from.ncols(), to.ncols());
    assert_eq!(result.len(), from.nrows());

    for (idx_from, row_from) in from.iter_rows().enumerate() {
        let mut min_dist = std::f64::MAX;
        let mut min_idx: usize = 0;
        for (idx_to, row_to) in to.iter_rows().enumerate() {
            let dist = EUCLIDEAN_SQ.distance(row_from, row_to);
            if dist < min_dist {
                min_dist = dist;
                min_idx = idx_to
            }
        }
        result[idx_from] = (min_idx, min_dist.sqrt());
    }
    result
}

/*
pub fn par_nearest_neighbor(from: &[f64], to: &DataFrame<f64>, num_threads: usize) -> (usize, f64) {
    assert_eq!(from.len(), to.ncols());
    thread::scope(|s| {
        let (tx, rx) = mpsc::channel();
        let data = to.data();

        let total_rows = to.nrows();
        let col_count = to.ncols();
        let rows_per_thread = total_rows / num_threads;
        let remainder = total_rows % num_threads;
        let mut done = 0;

        let mut threads = Vec::with_capacity(num_threads);

        for i in 0..num_threads {
            let mut rows_todo = rows_per_thread;
            if i < remainder {
                rows_todo += 1;
            }
            let tx1 = mpsc::Sender::clone(&tx);
            let start = done * col_count;
            let end = (done + rows_todo) * col_count;
            let slice = &data[start..end];

            let child = s.spawn(move |_| {
                let result = nearest_neighbor_slice(from, slice, done);
                tx1.send(result).unwrap();
            });

            threads.push(child);

            done += rows_todo;
        }

        let mut min_dist = std::f64::MAX;
        let mut min_idx: usize = 0;
        for _ in 0..num_threads {
            let (idx, dist) = rx.recv().unwrap();
            if dist < min_dist {
                min_dist = dist;
                min_idx = idx;
            }
        }
        (min_idx, min_dist)
    })
    .unwrap()
}

pub fn nearest_neighbor_slice(from: &[f64], to: &[f64], row_offset: usize) -> (usize, f64) {
    let num_cols = from.len();
    let mut min_dist = std::f64::MAX;
    let mut min_idx: usize = 0;
    for (idx_to, row_to) in to.chunks(num_cols).enumerate() {
        let dist = EUCLIDEAN_SQ.distance(from, row_to);
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx_to + row_offset
        }
    }
    (min_idx, min_dist.sqrt())
}
*/

#[cfg(test)]
mod test {
    use crate::calc::nn;
    use crate::data::DataFrame;
    use crate::map::som::Layer;
    use rand;
    use rand::Rng;

    #[test]
    fn xyf_nn() {
        let mut rng = rand::thread_rng();
        let from = [0.0, 0.0, 0.0, 0.0, 0.0];
        let mut to = DataFrame::empty(&["A", "B", "C", "D", "E"]);

        for _i in 0..10 {
            to.push_row(&[
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0, 2) as f64,
                rng.gen_range(0, 2) as f64,
            ]);
        }
        let layers = vec![Layer::cont(3, 0.5), Layer::cat(2, 0.5)];

        let (_idx, _dist) = nn::nearest_neighbor_xyf(&from, &to, &layers);
    }

    #[test]
    fn nn_simple() {
        let mut rng = rand::thread_rng();
        let from = [0.0, 0.0, 0.0];
        let mut to = DataFrame::empty(&["A", "B", "C"]);

        for _i in 0..100 {
            to.push_row(&[
                rng.gen_range(0.5, 1.0),
                rng.gen_range(0.5, 1.0),
                rng.gen_range(0.5, 1.0),
            ]);
        }
        to.push_row(&[0.0, 0.0, 0.2]);

        let (idx, _dist) = nn::nearest_neighbor(&from, &to);
        assert_eq!(idx, 100);
    }

    #[test]
    fn nns_simple() {
        let mut rng = rand::thread_rng();
        let mut from = DataFrame::empty(&["A", "B", "C"]);
        let mut to = DataFrame::empty(&["A", "B", "C"]);

        for _i in 0..100 {
            from.push_row(&[
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
            ]);
        }
        for _i in 0..100 {
            to.push_row(&[
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
                rng.gen_range(-1.0, 1.0),
            ]);
        }

        let result = vec![(0, 0.0); from.nrows()];
        let _result = nn::nearest_neighbors(&from, &to, result);

        //println!("{:?}", &result[0..20]);
    }
}
