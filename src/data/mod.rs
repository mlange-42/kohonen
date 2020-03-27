//! Data structures like tables.

use num_traits::Float;
use std::slice::{Chunks, ChunksMut};

#[allow(dead_code)]
pub struct DataFrame<T>
where
    T: Float,
{
    ncols: usize,
    nrows: usize,
    data: Vec<T>,
}

#[allow(dead_code)]
impl<T> DataFrame<T>
where
    T: Float,
{
    pub fn empty(ncols: usize) -> Self {
        DataFrame {
            ncols,
            nrows: 0,
            data: vec![],
        }
    }

    pub fn filled(nrows: usize, ncols: usize, fill: T) -> Self {
        DataFrame {
            ncols,
            nrows,
            data: vec![fill; nrows * ncols],
        }
    }

    pub fn from_rows(rows: &[Vec<T>]) -> Self {
        DataFrame {
            ncols: rows[0].len(),
            nrows: rows.len(),
            data: rows.iter().flatten().copied().collect(),
        }
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn push_row(&mut self, row: &[T]) {
        assert_eq!(row.len(), self.ncols);
        self.data.extend_from_slice(row);
        self.nrows += 1;
    }
    pub fn push_row_iter(&mut self, row: impl Iterator<Item = T>) {
        self.data.extend(row);
        self.nrows += 1;
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        let idx = self.index(row, col);
        &self.data[idx]
    }
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.index(row, col);
        &mut self.data[idx]
    }
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self.index(row, col);
        self.data[idx] = value
    }

    pub fn get_at(&self, index: usize) -> &T {
        &self.data[index]
    }
    pub fn get_mut_at(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
    pub fn set_at(&mut self, index: usize, value: T) {
        self.data[index] = value
    }

    pub fn get_row(&self, row: usize) -> &[T] {
        let idx = self.index(row, 0);
        &self.data[idx..idx + self.ncols]
    }
    pub fn get_row_mut(&mut self, row: usize, col: usize) -> &mut [T] {
        let idx = self.index(row, col);
        &mut self.data[idx..idx + self.ncols]
    }

    #[inline]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.ncols + col
    }

    #[inline]
    pub fn to_row_col(&self, index: usize) -> (usize, usize) {
        (index / self.ncols, index % self.ncols)
    }

    pub fn iter_rows(&self) -> Chunks<T> {
        self.data.chunks(self.ncols)
    }
    pub fn iter_rows_mut(&mut self) -> ChunksMut<T> {
        self.data.chunks_mut(self.ncols)
    }

    pub fn copy_column(&self, column: usize) -> Vec<T> {
        self.iter_rows().map(|row| row[column]).collect()
    }

    pub fn ranges(&self) -> Vec<(T, T)> {
        let ncol = self.ncols;
        let mut min = vec![T::max_value(); self.ncols];
        let mut max = vec![T::min_value(); self.ncols];
        for row in self.iter_rows() {
            for col in 0..ncol {
                let v = row[col];
                if v < min[col] {
                    min[col] = v;
                }
                if v > max[col] {
                    max[col] = v;
                }
            }
        }
        min.into_iter().zip(max).collect()
    }
    pub fn means(&self) -> Vec<T> {
        let ncol = self.ncols;
        let nrows = T::from(self.nrows).unwrap();

        let mut means = vec![T::zero(); ncol];
        for row in self.iter_rows() {
            for col in 0..ncol {
                let v = row[col];
                means[col] = means[col] + v;
            }
        }
        for col in 0..ncol {
            means[col] = means[col] / nrows;
        }
        means
    }
}

#[cfg(test)]
mod test {
    use crate::data::DataFrame;

    #[test]
    fn create_df() {
        let cols = 4;
        let rows = 100;
        let df = DataFrame::<f32>::filled(rows, cols, 0.0);

        assert_eq!(df.ncols, cols);
        assert_eq!(df.nrows, rows);
        assert_eq!(df.data.len(), rows * cols);
    }

    #[test]
    fn create_df_from_rows() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.],
            vec![3.0, 4.0, 5.0, 6.0],
        ];
        let df = DataFrame::<f32>::from_rows(&data);

        assert_eq!(df.ncols, 4);
        assert_eq!(df.nrows, 3);
        assert_eq!(df.get(1, 1), &3.0);
    }

    #[test]
    fn add_rows() {
        let cols = 4;
        let mut df = DataFrame::<f32>::empty(cols);

        df.push_row(&[1.0, 2.0, 3.0, 4.0]);
        df.push_row(&[2.0, 3.0, 4.0, 5.0]);
        df.push_row(&[3.0, 4.0, 5.0, 6.0]);

        assert_eq!(df.ncols, cols);
        assert_eq!(df.nrows, 3);
        assert_eq!(df.data.len(), 3 * cols);

        assert_eq!(df.get_row(1), &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(df.get(1, 2), &4.0);
        assert_eq!(df.get_at(2), &3.0);
    }

    #[test]
    fn iter_rows() {
        let cols = 4;
        let rows = 10;
        let mut df = DataFrame::<f32>::empty(cols);

        for _i in 0..rows {
            df.push_row(&[1.0, 2.0, 3.0, 4.0]);
        }

        let mut cnt = 0;
        for _row in df.iter_rows() {
            cnt += 1;
        }

        assert_eq!(cnt, rows);
    }

    #[test]
    fn ranges() {
        let cols = 4;
        let mut df = DataFrame::<f32>::empty(cols);

        df.push_row(&[1.0, 2.0, 3.0, 4.0]);
        df.push_row(&[2.0, 3.0, 4.0, 5.0]);
        df.push_row(&[3.0, 4.0, 5.0, 6.0]);

        let ranges = df.ranges();

        assert_eq!(ranges, vec![(1.0, 3.0), (2.0, 4.0), (3.0, 5.0), (4.0, 6.0)]);
    }
}
