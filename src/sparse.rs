// src/sparse.rs

//! A module for converting a dense matrix to a sparse storage format and performing operations.
//!
//! This implementation is a safe, idiomatic Rust translation of the `sprsin`, `sprsax`,
//! and `sprstx` routines from "Numerical Recipes".

/// Represents a square matrix in a sparse format.
///
/// This format, from "Numerical Recipes", stores the main diagonal first,
/// followed by all non-zero off-diagonal elements. A second vector, `ija`,
/// acts as a lookup table for row pointers and column indices.
#[derive(Debug, PartialEq)]
pub struct SparseMatrix {
    /// The dimension of the original square matrix.
    n: usize,
    /// `sa`: Stores the matrix values. The first `n` elements are the main
    /// diagonal. The subsequent elements are the off-diagonal non-zero values.
    sa: Vec<f64>,
    /// `ija`: The lookup vector.
    /// - `ija[0..n]` stores "row pointers". `ija[i]` is the index in `sa`
    ///   where the off-diagonal elements for row `i` begin.
    /// - `ija[n]` is a sentinel, one past the last off-diagonal element index.
    /// - `ija[(n+1)..]` stores the column indices for the corresponding off-diagonal
    ///   values in `sa`.
    ija: Vec<usize>,
}

impl SparseMatrix {
    /// Creates a `SparseMatrix` from a dense, square matrix.
    pub fn from_dense(
        dense_matrix: &[Vec<f64>],
        threshold: f64,
    ) -> Result<Self, &'static str> {
        let n = dense_matrix.len();
        if n > 0 && dense_matrix.iter().any(|row| row.len() != n) {
            return Err("Input matrix must be square.");
        }

        let mut sa = Vec::with_capacity(n * 2);
        let mut ija = Vec::with_capacity(n * 2);

        for i in 0..n {
            sa.push(dense_matrix[i][i]);
        }
        
        ija.resize(n + 1, 0);

        for i in 0..n {
            ija[i] = sa.len();
            for j in 0..n {
                let value = dense_matrix[i][j];
                if i != j && value.abs() >= threshold {
                    sa.push(value);
                    ija.push(j);
                }
            }
        }
        ija[n] = sa.len();

        Ok(SparseMatrix { n, sa, ija })
    }

    /// Retrieves the value at a given (row, col) from the sparse matrix.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n || col >= self.n {
            return 0.0;
        }
        if row == col {
            return self.sa[row];
        }
        let row_start_idx = self.ija[row];
        let row_end_idx = self.ija[row + 1];
        let col_data_start = self.n + 1;
        let start_offset = row_start_idx - self.n;
        
        for i in 0..(row_end_idx - row_start_idx) {
            if self.ija[col_data_start + start_offset + i] == col {
                return self.sa[row_start_idx + i];
            }
        }
        0.0
    }

    /// Performs a matrix-vector multiplication: `b = A * x`.
    pub fn multiply_vector(&self, x: &[f64]) -> Result<Vec<f64>, &'static str> {
        if x.len() != self.n {
            return Err("Vector length must match matrix dimension.");
        }

        let mut b = vec![0.0; self.n];
        let col_data_start = self.n + 1;

        for i in 0..self.n {
            b[i] = self.sa[i] * x[i];
            
            let row_start_idx = self.ija[i];
            let row_end_idx = self.ija[i + 1];
            let start_offset = row_start_idx - self.n;
            
            for k in 0..(row_end_idx - row_start_idx) {
                let val_idx = row_start_idx + k;
                let col_idx = self.ija[col_data_start + start_offset + k];
                b[i] += self.sa[val_idx] * x[col_idx];
            }
        }
        Ok(b)
    }

    /// Performs a transpose-matrix-vector multiplication: `b = A^T * x`.
    pub fn multiply_vector_transpose(&self, x: &[f64]) -> Result<Vec<f64>, &'static str> {
        if x.len() != self.n {
            return Err("Vector length must match matrix dimension.");
        }

        // Initialize result with the diagonal part of the operation.
        let mut b: Vec<f64> = self.sa.iter().take(self.n).zip(x.iter()).map(|(diag, xi)| diag * xi).collect();
        let col_data_start = self.n + 1;

        // Add the off-diagonal parts.
        for i in 0..self.n { // For each row `i` of the original matrix A...
            let row_start_idx = self.ija[i];
            let row_end_idx = self.ija[i + 1];
            let start_offset = row_start_idx - self.n;

            for k in 0..(row_end_idx - row_start_idx) {
                let val_idx = row_start_idx + k;
                let col_idx = self.ija[col_data_start + start_offset + k]; // This is `j` in A[i][j]
                b[col_idx] += self.sa[val_idx] * x[i];
            }
        }
        Ok(b)
    }

    /// Computes the transpose of the sparse matrix, returning a new `SparseMatrix`.
    /// This is a safe, idiomatic implementation of `sprstp`.
    pub fn transpose(&self) -> Self {
        let n = self.n;
        let mut new_sa = vec![0.0; n];
        let mut new_ija = vec![0; n + 1];
        
        // The new diagonal is the same as the old one.
        new_sa[..n].copy_from_slice(&self.sa[..n]);

        // Temporary structure to hold the unsorted off-diagonal elements of the transpose.
        // `transposed_rows[j]` will hold all `(i, A[i][j])` pairs.
        let mut transposed_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let col_data_start = self.n + 1;

        // Populate the temporary structure.
        for i in 0..n { // For each row `i` of the original matrix...
            let row_start_idx = self.ija[i];
            let row_end_idx = self.ija[i + 1];
            let start_offset = row_start_idx - self.n;

            for k in 0..(row_end_idx - row_start_idx) {
                let val_idx = row_start_idx + k;
                let col_idx = self.ija[col_data_start + start_offset + k]; // This is `j`
                // In the transpose, A[i][j] becomes A_T[j][i].
                transposed_rows[col_idx].push((i, self.sa[val_idx]));
            }
        }
        
        // Build the final `sa` and `ija` from the sorted temporary structure.
        for j in 0..n { // For each row `j` of the new transposed matrix...
            new_ija[j] = new_sa.len();
            transposed_rows[j].sort_unstable_by_key(|&(col, _)| col);
            for &(col, val) in &transposed_rows[j] {
                new_sa.push(val);
                new_ija.push(col);
            }
        }
        new_ija[n] = new_sa.len();

        SparseMatrix { n, sa: new_sa, ija: new_ija }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> SparseMatrix {
        let dense = vec![
            vec![5.0, 0.0, 2.0],
            vec![3.0, 9.0, 0.001],
            vec![0.0, 1.0, 7.0],
        ];
        SparseMatrix::from_dense(&dense, 0.1).unwrap()
    }

    #[test]
    fn test_sprsin_conversion() {
        let sparse = create_test_matrix();
        let expected_sa = vec![5.0, 9.0, 7.0, 2.0, 3.0, 1.0];
        let expected_ija = vec![3, 4, 5, 6, 2, 0, 1];
        assert_eq!(sparse.sa, expected_sa);
        assert_eq!(sparse.ija, expected_ija);
    }

    #[test]
    fn test_get_method() {
        let sparse = create_test_matrix();
        assert_eq!(sparse.get(0, 0), 5.0);
        assert_eq!(sparse.get(0, 2), 2.0);
        assert_eq!(sparse.get(1, 0), 3.0);
        assert_eq!(sparse.get(0, 1), 0.0);
    }

    #[test]
    fn test_multiply_vector() {
        let sparse = create_test_matrix();
        let x = vec![1.0, 2.0, 3.0];
        let expected_b = vec![11.0, 21.0, 23.0];
        let b = sparse.multiply_vector(&x).unwrap();
        b.iter().zip(expected_b.iter()).for_each(|(val, expected)| {
            assert!((val - expected).abs() < 1e-10);
        });
    }

    #[test]
    fn test_multiply_vector_transpose() {
        // A^T is:
        // [5, 3, 0]
        // [0, 9, 1]
        // [2, 0, 7]
        let sparse = create_test_matrix();
        let x = vec![1.0, 2.0, 3.0];

        // Manually calculate b = A^T * x
        // b[0] = 5*1 + 3*2 + 0*3 = 11
        // b[1] = 0*1 + 9*2 + 1*3 = 21
        // b[2] = 2*1 + 0*2 + 7*3 = 23
        let expected_b = vec![11.0, 21.0, 23.0];
        
        let b = sparse.multiply_vector_transpose(&x).unwrap();
        b.iter().zip(expected_b.iter()).for_each(|(val, expected)| {
            assert!((val - expected).abs() < 1e-10);
        });
    }
    
    #[test]
    fn test_transpose_method() {
        let sparse = create_test_matrix();
        let transposed = sparse.transpose();
        
        // Original matrix had A[0][2] = 2.0
        // Transposed matrix should have A_T[2][0] = 2.0
        assert_eq!(sparse.get(0, 2), 2.0);
        assert_eq!(transposed.get(2, 0), 2.0);
        
        // Original matrix had A[2][1] = 1.0
        // Transposed matrix should have A_T[1][2] = 1.0
        assert_eq!(sparse.get(2, 1), 1.0);
        assert_eq!(transposed.get(1, 2), 1.0);
        
        // Check a zero value that should remain zero
        assert_eq!(sparse.get(0, 1), 0.0);
        assert_eq!(transposed.get(1, 0), 0.0);

        // Check the diagonals (should be the same)
        assert_eq!(sparse.get(1, 1), transposed.get(1, 1));
    }

    #[test]
    fn test_non_square_matrix_error() {
        let dense = vec![vec![1.0, 2.0]];
        assert!(SparseMatrix::from_dense(&dense, 0.0).is_err());
    }
}
