use ndarray::prelude::*;

pub fn banbks(
    a: &Array2<f64>,
    m1: usize,
    m2: usize,
    al: &Array2<f64>,
    indx: &Array1<usize>,
    b: &mut Array1<f64>,
) {
    let n = a.nrows();
    let mm = m1 + m2 + 1;
    let mut l = m1;

    // Forward substitution
    for k in 0..n {
        let pivot_row = indx[k];
        if pivot_row != k {
            b.swap(k, pivot_row);
        }
        
        if k < n - 1 {
            l = l.min(n - k - 1);
            for i in 1..=l {
                let j = k + i;
                if j < n {
                    b[j] -= al[[k, i]] * b[k];
                }
            }
        }
    }

    // Back substitution
    l = 1;
    for i in (0..n).rev() {
        let mut sum = b[i];
        if i < n - 1 {
            let k_max = l.min(n - i - 1);
            for k in 1..=k_max {
                let j = i + k;
                if j < n && m1 + k < a.ncols() {
                    sum -= a[[i, m1 + k]] * b[j];
                }
            }
        }
        if m1 < a.ncols() {
            b[i] = sum / a[[i, m1]];
        }
        if l < mm {
            l += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_tridiagonal_system() {
        let m1 = 1;  // lower bandwidth
        let m2 = 1;  // upper bandwidth

        // Matrix: [2, 1, 0]
        //         [1, 2, 1] 
        //         [0, 1, 2]
        let a = arr2(&[
            [0.0, 2.0, 1.0],  // row 0
            [1.0, 2.0, 1.0],  // row 1
            [1.0, 2.0, 0.0],  // row 2
        ]);

        // Lower triangular multipliers
        let al = arr2(&[
            [0.0, 0.0, 0.0],      // row 0
            [0.0, 0.5, 0.0],      // row 1
            [0.0, 2.0/3.0, 0.0],  // row 2
        ]);

        // Pivot indices (no pivoting)
        let indx = arr1(&[0, 1, 2]);

        // Right-hand side: A * [1, 2, 2] = [4, 8, 6]
        let mut b = arr1(&[4.0, 8.0, 6.0]);

        banbks(&a, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2, 2]
        assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simple_2x2_system() {
        let m1 = 1;
        let m2 = 1;

        // Matrix: [4, 1]
        //         [1, 4]
        let a = arr2(&[
            [0.0, 4.0, 1.0],  // row 0
            [1.0, 4.0, 0.0],  // row 1
        ]);

        // Multipliers
        let al = arr2(&[
            [0.0, 0.0, 0.0],  // row 0
            [0.0, 0.25, 0.0], // row 1
        ]);

        let indx = arr1(&[0, 1]);
        // RHS: [4*1 + 1*2 = 6, 1*1 + 4*2 = 9]
        let mut b = arr1(&[6.0, 9.0]);

        banbks(&a, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2]
        assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_system() {
        let m1 = 0;  // no lower diagonal
        let m2 = 0;  // no upper diagonal

        // Diagonal matrix: [2, 0, 0]
        //                 [0, 3, 0]
        //                 [0, 0, 4]
        let a = arr2(&[
            [2.0],  // row 0
            [3.0],  // row 1
            [4.0],  // row 2
        ]);

        // No multipliers
        let al = arr2(&[
            [0.0],  // row 0
            [0.0],  // row 1
            [0.0],  // row 2
        ]);

        let indx = arr1(&[0, 1, 2]);
        // RHS: [2*1=2, 3*2=6, 4*3=12]
        let mut b = arr1(&[2.0, 6.0, 12.0]);

        banbks(&a, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2, 3]
        assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_banded_system_with_pivoting() {
        let m1 = 1;
        let m2 = 1;

        // Matrix that requires pivoting
        let a = arr2(&[
            [0.0, 0.0, 1.0],  // row 0 - will be pivoted
            [1.0, 2.0, 1.0],  // row 1
            [1.0, 2.0, 0.0],  // row 2
        ]);

        // Multipliers after pivoting
        let al = arr2(&[
            [0.0, 0.0, 0.0],  // row 0
            [0.0, 0.5, 0.0],  // row 1
            [0.0, 2.0/3.0, 0.0], // row 2
        ]);

        // Pivot indices indicating row 0 and 1 were swapped
        let indx = arr1(&[1, 0, 2]);

        let mut b = arr1(&[1.0, 8.0, 6.0]);

        banbks(&a, m1, m2, &al, &indx, &mut b);

        // Should get valid solution
        assert!(!b[0].is_nan());
        assert!(!b[1].is_nan());
        assert!(!b[2].is_nan());
    }

    #[test]
    fn test_larger_banded_system() {
        let m1 = 2;
        let m2 = 1;

        // 4x4 banded matrix
        let a = arr2(&[
            [0.0, 0.0, 4.0, 1.0],  // row 0
            [0.0, 3.0, 5.0, 2.0],  // row 1
            [1.0, 2.0, 6.0, 3.0],  // row 2
            [2.0, 1.0, 7.0, 0.0],  // row 3
        ]);

        // Example multipliers
        let al = arr2(&[
            [0.0, 0.0, 0.0, 0.0],  // row 0
            [0.0, 0.0, 0.75, 0.0], // row 1
            [0.0, 0.0, 0.4, 0.0],  // row 2
            [0.0, 0.0, 0.3, 0.0],  // row 3
        ]);

        let indx = arr1(&[0, 1, 2, 3]);
        
        // Create a known solution and compute RHS
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let mut b = Array1::zeros(4);
        
        // Compute A*x manually
        for i in 0..4 {
            for j in 0..4 {
                if i <= j + m1 && j <= i + m2 {
                    let col_index = m1 + j - i;
                    if col_index < a.ncols() {
                        b[i] += a[[i, col_index]] * x[j];
                    }
                }
            }
        }

        let mut b_solve = b.clone();
        banbks(&a, m1, m2, &al, &indx, &mut b_solve);

        // Check if we recover the original solution
        for i in 0..4 {
            assert_abs_diff_eq!(b_solve[i], x[i], epsilon = 1e-10);
        }
    }
}
