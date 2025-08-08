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
}
