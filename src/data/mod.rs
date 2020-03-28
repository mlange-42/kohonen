//! Data structures like tables.

use std::slice::{Chunks, ChunksMut};

/// A data frame with all columns of the same Float type.
#[allow(dead_code)]
pub struct DataFrame {
    ncols: usize,
    nrows: usize,
    names: Vec<String>,
    data: Vec<f64>,
}

#[allow(dead_code)]
impl DataFrame {
    /// Creates an empty data frame, with the given columns and zero rows.
    pub fn empty(columns: &[&str]) -> Self {
        DataFrame {
            names: columns.iter().map(|s| s.to_string()).collect(),
            ncols: columns.len(),
            nrows: 0,
            data: vec![],
        }
    }

    /// Creates a blank data frame, with the given number of columns and rows, filled with a value.
    pub fn filled(nrows: usize, columns: &[&str], fill: f64) -> Self {
        DataFrame {
            names: columns.iter().map(|s| s.to_string()).collect(),
            ncols: columns.len(),
            nrows,
            data: vec![fill; nrows * columns.len()],
        }
    }

    /// Creates a data frame from a vector of rows.
    pub fn from_rows(columns: &[&str], rows: &[Vec<f64>]) -> Self {
        assert_eq!(columns.len(), rows[0].len());
        DataFrame {
            names: columns.iter().map(|s| s.to_string()).collect(),
            ncols: rows[0].len(),
            nrows: rows.len(),
            data: rows.iter().flatten().copied().collect(),
        }
    }

    /// Number of columns in the data frame.
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Number of rows in the data frame.
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// A reference to the raw data: a flat vector of values in row-first order.
    ///
    /// Example:
    ///
    /// ` x x x x x x x x x x x x x x x ...`
    ///
    /// `|___ row 1 ___|___ row 2 ___|___ ...`
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a reference to the data frame's column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Appends a row to the end of the data frame, from a slice.
    pub fn push_row(&mut self, row: &[f64]) {
        assert_eq!(row.len(), self.ncols);
        self.data.extend_from_slice(row);
        self.nrows += 1;
    }
    /// Appends a row to the end of the data frame, from an iterator.
    pub fn push_row_iter(&mut self, row: impl Iterator<Item = f64>) {
        self.data.extend(row);
        self.nrows += 1;
    }
    /// Returns a reference to the value at (row, column).
    pub fn get(&self, row: usize, col: usize) -> &f64 {
        let idx = self.index(row, col);
        &self.data[idx]
    }
    /// Returns a mutable reference to the value at (row, column).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        let idx = self.index(row, col);
        &mut self.data[idx]
    }
    /// Sets the value at (row, column), consuming the value.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.index(row, col);
        self.data[idx] = value
    }

    /// Returns a reference to the value at the given index in raw data.
    pub fn get_at(&self, index: usize) -> &f64 {
        &self.data[index]
    }
    /// Returns a mutable reference to the value at the given index in raw data.
    pub fn get_mut_at(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index]
    }
    /// Sets the value at the given index in raw data, consuming the value.
    pub fn set_at(&mut self, index: usize, value: f64) {
        self.data[index] = value
    }

    /// Returns a row as a slice reference.
    pub fn get_row(&self, row: usize) -> &[f64] {
        let idx = self.index(row, 0);
        &self.data[idx..idx + self.ncols]
    }
    /// Returns a row as a mutable slice reference.
    pub fn get_row_mut(&mut self, row: usize, col: usize) -> &mut [f64] {
        let idx = self.index(row, col);
        &mut self.data[idx..idx + self.ncols]
    }

    /// Returns the raw data index for (row, col).
    #[inline]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.ncols + col
    }

    /// Returns (row, col) for a raw data index.
    #[inline]
    pub fn to_row_col(&self, index: usize) -> (usize, usize) {
        (index / self.ncols, index % self.ncols)
    }

    /// An iterator over rows.
    pub fn iter_rows(&self) -> Chunks<f64> {
        self.data.chunks(self.ncols)
    }
    /// A mutable iterator over rows.
    pub fn iter_rows_mut(&mut self) -> ChunksMut<f64> {
        self.data.chunks_mut(self.ncols)
    }

    /// Copies a column's values into a new vector.
    pub fn copy_column(&self, column: usize) -> Vec<f64> {
        self.iter_rows().map(|row| row[column]).collect()
    }

    /// Returns ranges of columns.
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        let ncol = self.ncols;
        let mut min = vec![std::f64::MAX; ncol];
        let mut max = vec![std::f64::MIN; ncol];
        let mut any = vec![false; ncol];
        for row in self.iter_rows() {
            for col in 0..ncol {
                let v = row[col];
                if !v.is_nan() {
                    if v < min[col] {
                        min[col] = v;
                    }
                    if v > max[col] {
                        max[col] = v;
                    }
                    any[col] = true;
                }
            }
        }
        min.into_iter()
            .zip(max)
            .enumerate()
            .map(|(i, (mn, mx))| {
                if any[i] {
                    (mn, mx)
                } else {
                    (std::f64::NAN, std::f64::NAN)
                }
            })
            .collect()
    }

    /// Returns means of columns.
    pub fn means(&self) -> Vec<f64> {
        let ncol = self.ncols;
        //let nrows = self.nrows;

        let mut means = vec![0.0; ncol];
        let mut counts = vec![0; ncol];
        for row in self.iter_rows() {
            for col in 0..ncol {
                let v = row[col];
                if !v.is_nan() {
                    means[col] += v;
                    counts[col] += 1;
                }
            }
        }
        for i in 0..ncol {
            means[i] /= counts[i] as f64;
        }
        means
    }
}

#[cfg(test)]
mod test {
    use crate::data::DataFrame;

    #[test]
    fn create_df() {
        let cols = ["A", "B", "C", "D"];
        let rows = 100;
        let df = DataFrame::filled(rows, &cols, 0.0);

        assert_eq!(df.ncols, cols.len());
        assert_eq!(df.nrows, rows);
        assert_eq!(df.data.len(), rows * cols.len());
    }

    #[test]
    fn create_df_from_rows() {
        let cols = ["A", "B", "C", "D"];
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.],
            vec![3.0, 4.0, 5.0, 6.0],
        ];
        let df = DataFrame::from_rows(&cols, &data);

        assert_eq!(df.ncols, 4);
        assert_eq!(df.nrows, 3);
        assert_eq!(df.get(1, 1), &3.0);
    }

    #[test]
    fn add_rows() {
        let cols = ["A", "B", "C", "D"];
        let mut df = DataFrame::empty(&cols);

        df.push_row(&[1.0, 2.0, 3.0, 4.0]);
        df.push_row(&[2.0, 3.0, 4.0, 5.0]);
        df.push_row(&[3.0, 4.0, 5.0, 6.0]);

        assert_eq!(df.ncols, cols.len());
        assert_eq!(df.nrows, 3);
        assert_eq!(df.data.len(), 3 * cols.len());

        assert_eq!(df.get_row(1), &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(df.get(1, 2), &4.0);
        assert_eq!(df.get_at(2), &3.0);
    }

    #[test]
    fn iter_rows() {
        let cols = ["A", "B", "C", "D"];
        let rows = 10;
        let mut df = DataFrame::empty(&cols);

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
        let cols = ["A", "B", "C", "D"];
        let mut df = DataFrame::empty(&cols);

        df.push_row(&[1.0, 2.0, 3.0, 4.0]);
        df.push_row(&[2.0, 3.0, 4.0, 5.0]);
        df.push_row(&[3.0, 4.0, 5.0, 6.0]);

        let ranges = df.ranges();

        assert_eq!(ranges, vec![(1.0, 3.0), (2.0, 4.0), (3.0, 5.0), (4.0, 6.0)]);
    }
}
