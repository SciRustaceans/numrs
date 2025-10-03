// src/lu_decomp.rs

//! An optimized, multithreaded implementation of banded LU decomposition.

use rayon::prelude::*;

const TINY: f64 = 1.0e-40; // Use a smaller TINY for floating point safety

/// Represents a matrix in compact banded format.
#[derive(Debug, Clone)]
pub struct BandedMatrix {
    pub n: usize,
    pub m1: usize,
    pub m2: usize,
    data: Vec<Vec<f64>>,
}

/// Represents the result of an LU decomposition.
pub struct LUDecomposedMatrix {
    matrix: BandedMatrix,
    pivot_indices: Vec<usize>,
    determinant_sign: f64,
}

impl BandedMatrix {
    /// Creates a new BandedMatrix, initialized to zeros.
    pub fn new(n: usize, m1: usize, m2: usize) -> Self {
        let band_width = m1 + m2 + 1;
        BandedMatrix {
            n, m1, m2,
            data: vec![vec![0.0; band_width]; n],
        }
    }

    /// Gets a mutable reference to the element at logical position (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        let band_col = col as isize - row as isize + self.m1 as isize;
        if band_col >= 0 && band_col < (self.m1 + self.m2 + 1) as isize {
            self.data.get_mut(row).and_then(|r| r.get_mut(band_col as usize))
        } else {
            None // Out of band
        }
    }

    /// Performs LU decomposition with partial pivoting.
    pub fn lu_decompose(mut self) -> Result<LUDecomposedMatrix, &'static str> {
        let mut pivot_indices = vec![0; self.n];
        let mut determinant_sign = 1.0;
        let band_width = self.m1 + self.m2 + 1;

        for k in 0..self.n {
            let mut pivot_val = self.data[k][self.m1];
            let mut pivot_row = k;
            
            let search_limit = self.n.min(k + self.m1 + 1);
            for i in (k + 1)..search_limit {
                if self.data[i][self.m1 - (i - k)].abs() > pivot_val.abs() {
                    pivot_val = self.data[i][self.m1 - (i - k)];
                    pivot_row = i;
                }
            }
            pivot_indices[k] = pivot_row;

            if pivot_row != k {
                self.data.swap(k, pivot_row);
                determinant_sign = -determinant_sign;
            }
            
            if self.data[k][self.m1].abs() < TINY {
                 return Err("Singular matrix encountered.");
            }

            let pivot_value = self.data[k][self.m1];
            let pivot_row_slice: Vec<f64> = self.data[k][(self.m1 + 1)..band_width].to_vec();

            let lower_limit = self.n.min(k + self.m1 + 1);
            self.data[(k + 1)..lower_limit]
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, row_i)| {
                    let i = k + 1 + idx;
                    let j_k = self.m1 - (i - k);
                    
                    let pivot_factor = row_i[j_k] / pivot_value;
                    row_i[j_k] = pivot_factor;

                    for j_offset in 0..pivot_row_slice.len() {
                        if (j_k + 1 + j_offset) < row_i.len() {
                             row_i[j_k + 1 + j_offset] -= pivot_factor * pivot_row_slice[j_offset];
                        }
                    }
                });
        }

        Ok(LUDecomposedMatrix {
            matrix: self,
            pivot_indices,
            determinant_sign,
        })
    }
}

impl LUDecomposedMatrix {
    /// Solves the system Ax = b using the decomposed matrix.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.matrix.n;
        let m1 = self.matrix.m1;
        let mut x = b.to_vec();

        for k in 0..n {
            let pivot_row = self.pivot_indices[k];
            if pivot_row != k {
                x.swap(k, pivot_row);
            }
            let search_limit = n.min(k + m1 + 1);
            for i in (k + 1)..search_limit {
                x[i] -= self.matrix.data[i][m1 - (i - k)] * x[k];
            }
        }

        for i in (0..n).rev() {
            let mut sum = x[i];
            let search_limit = n.min(i + self.matrix.m2 + 1);
            for j in (i + 1)..search_limit {
                sum -= self.matrix.data[i][m1 + j - i] * x[j];
            }
            x[i] = sum / self.matrix.data[i][m1];
        }
        x
    }
    
    /// Calculates the determinant of the original matrix.
    /// This uses the `determinant_sign` field, resolving the dead_code warning.
    pub fn determinant(&self) -> f64 {
        let mut det = self.determinant_sign;
        for i in 0..self.matrix.n {
            // The diagonal elements of the U matrix are stored on the main diagonal
            // of the decomposed data.
            det *= self.matrix.data[i][self.matrix.m1];
        }
        det
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu_decomposition_solve_and_determinant() {
        // Create a 4x4 banded matrix with m1=1, m2=1 (tridiagonal)
        let n = 4;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Populate the matrix A
        //  2 -1  0  0
        // -1  2 -1  0
        //  0 -1  2 -1
        //  0  0 -1  2
        *a.get_mut(0, 0).unwrap() = 2.0; *a.get_mut(0, 1).unwrap() = -1.0;
        *a.get_mut(1, 0).unwrap() = -1.0; *a.get_mut(1, 1).unwrap() = 2.0; *a.get_mut(1, 2).unwrap() = -1.0;
        *a.get_mut(2, 1).unwrap() = -1.0; *a.get_mut(2, 2).unwrap() = 2.0; *a.get_mut(2, 3).unwrap() = -1.0;
        *a.get_mut(3, 2).unwrap() = -1.0; *a.get_mut(3, 3).unwrap() = 2.0;

        // Define the right-hand side vector b for the system Ax = b
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let expected_x = vec![2.0, 3.0, 3.0, 2.0];
        let expected_determinant = 5.0; // Determinant of this specific matrix is 5.

        // 1. Decompose the matrix
        let lu = a.lu_decompose().expect("Decomposition failed");

        // 2. Solve the system
        let x = lu.solve(&b);
        
        // 3. Calculate the determinant
        let det = lu.determinant();

        // 4. Assert all results
        assert!((det - expected_determinant).abs() < 1e-10, "Determinant is incorrect");
        for i in 0..n {
            assert!((x[i] - expected_x[i]).abs() < 1e-10, "Solution differs at index {}", i);
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let n = 3;
        let m1 = 0;
        let m2 = 0;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a diagonal matrix
        *a.get_mut(0, 0).unwrap() = 2.0;
        *a.get_mut(1, 1).unwrap() = 3.0;
        *a.get_mut(2, 2).unwrap() = 4.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![4.0, 9.0, 12.0];
        let x = lu.solve(&b);

        assert!((lu.determinant() - 24.0).abs() < 1e-10);
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a singular matrix (rows are linearly dependent)
        *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 1.0;
        *a.get_mut(1, 0).unwrap() = 1.0; *a.get_mut(1, 1).unwrap() = 1.0; *a.get_mut(1, 2).unwrap() = 1.0;
        *a.get_mut(2, 1).unwrap() = 1.0; *a.get_mut(2, 2).unwrap() = 1.0;

        let result = a.lu_decompose();
        assert!(result.is_err());
    }

    #[test]
    fn test_ill_conditioned_matrix() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a matrix with very small diagonal element
        *a.get_mut(0, 0).unwrap() = 1e-20; *a.get_mut(0, 1).unwrap() = 1.0;
        *a.get_mut(1, 0).unwrap() = 1.0; *a.get_mut(1, 1).unwrap() = 2.0; *a.get_mut(1, 2).unwrap() = 1.0;
        *a.get_mut(2, 1).unwrap() = 1.0; *a.get_mut(2, 2).unwrap() = 2.0;

        let result = a.lu_decompose();
        // Should handle small diagonal elements with pivoting
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_matrix() {
        let n = 100;
        let m1 = 2;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a larger banded matrix
        for i in 0..n {
            *a.get_mut(i, i).unwrap() = 4.0;  // Diagonal
            if i > 0 { *a.get_mut(i, i - 1).unwrap() = -1.0; }  // Subdiagonal
            if i > 1 { *a.get_mut(i, i - 2).unwrap() = 0.5; }  // Second subdiagonal
            if i < n - 1 { *a.get_mut(i, i + 1).unwrap() = -1.0; }  // Superdiagonal
        }

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0; n];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite
        assert!(lu.determinant().is_finite());
    }

    #[test]
    fn test_matrix_with_zeros() {
        let n = 4;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create matrix with some zero elements
        *a.get_mut(0, 0).unwrap() = 2.0; *a.get_mut(0, 1).unwrap() = 0.0;
        *a.get_mut(1, 0).unwrap() = 0.0; *a.get_mut(1, 1).unwrap() = 3.0; *a.get_mut(1, 2).unwrap() = 1.0;
        *a.get_mut(2, 1).unwrap() = 1.0; *a.get_mut(2, 2).unwrap() = 0.0; *a.get_mut(2, 3).unwrap() = 2.0;
        *a.get_mut(3, 2).unwrap() = 1.0; *a.get_mut(3, 3).unwrap() = 1.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite
        assert!(lu.determinant().is_finite());
    }

    #[test]
    fn test_tridiagonal_symmetric_matrix() {
        let n = 5;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a symmetric tridiagonal matrix
        for i in 0..n {
            *a.get_mut(i, i).unwrap() = 2.0;  // Diagonal
            if i > 0 { *a.get_mut(i, i - 1).unwrap() = -1.0; }      // Subdiagonal
            if i < n - 1 { *a.get_mut(i, i + 1).unwrap() = -1.0; }  // Superdiagonal
        }

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0; n];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite and positive for this matrix
        let det = lu.determinant();
        assert!(det.is_finite());
        assert!(det > 0.0);
    }

    #[test]
    fn test_matrix_with_pivoting() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create a matrix that requires pivoting: small diagonal element
        *a.get_mut(0, 0).unwrap() = 1e-15; *a.get_mut(0, 1).unwrap() = 1.0;
        *a.get_mut(1, 0).unwrap() = 1.0; *a.get_mut(1, 1).unwrap() = 1.0; *a.get_mut(1, 2).unwrap() = 1.0;
        *a.get_mut(2, 1).unwrap() = 1.0; *a.get_mut(2, 2).unwrap() = 1.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0, 2.0, 1.0];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite
        assert!(lu.determinant().is_finite());
    }

    #[test]
    fn test_very_small_matrix() {
        let n = 1;
        let m1 = 0;
        let m2 = 0;
        let mut a = BandedMatrix::new(n, m1, m2);

        *a.get_mut(0, 0).unwrap() = 5.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![10.0];
        let x = lu.solve(&b);

        assert!((lu.determinant() - 5.0).abs() < 1e-10);
        assert!((x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_with_large_values() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create matrix with very large values
        *a.get_mut(0, 0).unwrap() = 1e10; *a.get_mut(0, 1).unwrap() = 1e5;
        *a.get_mut(1, 0).unwrap() = 1e5; *a.get_mut(1, 1).unwrap() = 1e10; *a.get_mut(1, 2).unwrap() = 1e5;
        *a.get_mut(2, 1).unwrap() = 1e5; *a.get_mut(2, 2).unwrap() = 1e10;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1e10, 2e10, 1e10];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite
        assert!(lu.determinant().is_finite());
    }

    #[test]
    fn test_matrix_with_negative_determinant() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create matrix with negative determinant (due to row swaps)
        *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 2.0;
        *a.get_mut(1, 0).unwrap() = 3.0; *a.get_mut(1, 1).unwrap() = 4.0; *a.get_mut(1, 2).unwrap() = 1.0;
        *a.get_mut(2, 1).unwrap() = 1.0; *a.get_mut(2, 2).unwrap() = 1.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0, 2.0, 1.0];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Determinant might be negative due to pivoting
        assert!(lu.determinant().is_finite());
    }

    #[test]
    fn test_matrix_with_alternating_signs() {
        let n = 4;
        let m1 = 1;
        let m2 = 1;
        let mut a = BandedMatrix::new(n, m1, m2);

        // Create matrix with alternating signs
        *a.get_mut(0, 0).unwrap() = 2.0; *a.get_mut(0, 1).unwrap() = -1.0;
        *a.get_mut(1, 0).unwrap() = -1.0; *a.get_mut(1, 1).unwrap() = 2.0; *a.get_mut(1, 2).unwrap() = -1.0;
        *a.get_mut(2, 1).unwrap() = -1.0; *a.get_mut(2, 2).unwrap() = 2.0; *a.get_mut(2, 3).unwrap() = -1.0;
        *a.get_mut(3, 2).unwrap() = -1.0; *a.get_mut(3, 3).unwrap() = 2.0;

        let lu = a.lu_decompose().expect("Decomposition failed");
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let x = lu.solve(&b);

        // Check that solution is finite
        for val in &x {
            assert!(val.is_finite());
        }
        
        // Check determinant is finite
        assert!(lu.determinant().is_finite());
    }
}
