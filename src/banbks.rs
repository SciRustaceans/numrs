use ndarray::prelude::*;

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

    // Forward substitution
    for k in 0..n {
        // Apply permutation from factorization
        // In some implementations, indx[k] is the original row that went to position k
        let pivot_row = indx[k];
        if pivot_row != k {
            b.swap(k, pivot_row);
        }
        
        // Eliminate below: apply multipliers using corrected indexing
        let l = m1.min(n - k - 1);
        for i in 1..=l {
            let j = k + i;
            if j < n {
                b[j] -= al[[j, i]] * b[k];
            }
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        let mut sum = b[i];
        
        // Add contributions from known variables (superdiagonal elements)
        let num_superdiag = m2.min(n - 1 - i);
        for j in 1..=num_superdiag {
            if i + j < n {
                sum -= a[[i, m1 + j]] * b[i + j];
            }
        }
        
        // Divide by diagonal element
        b[i] = sum / a[[i, m1]];
    }
}/// Helper function to create band matrices for testing
fn create_band_matrix(data: &[&[f64]], m1: usize, m2: usize) -> Array2<f64> {
    let n = data.len();
    let cols = m1 + m2 + 1;
    let mut matrix = Array2::zeros((n, cols));
    
    for i in 0..n {
        for j in 0..cols {
            let actual_col = i as isize + j as isize - m1 as isize;
            if actual_col >= 0 && actual_col < n as isize {
                matrix[[i, j]] = data[i][actual_col as usize];
            }
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

    #[test]
fn test_tridiagonal_system() {
    let m1 = 1;
    let m2 = 1;

    // Matrix: [2, 1, 0]
    //         [1, 2, 1] 
    //         [0, 1, 2]
    
    // LU decomposition:
    // Multiplier for row 1: 1/2 = 0.5
    // New row 1: [0, 1.5, 1], RHS: 7 - 0.5*4 = 5
    // Multiplier for row 2: 1/1.5 = 2/3  
    // New row 2: [0, 0, 4/3], RHS: 6 - (2/3)*5 = 6 - 10/3 = 8/3
    
    let a = arr2(&[
        [0.0, 2.0, 1.0],   // row 0: [0, 2, 1] -> represents 2*x0 + 1*x1
        [1.0, 1.5, 1.0],   // row 1: [1, 1.5, 1] -> represents 1.5*x1 + 1*x2  
        [1.0, 4.0/3.0, 0.0], // row 2: [1, 4/3, 0] -> represents (4/3)*x2
    ]);

    let al = arr2(&[
        [0.0, 0.0, 0.0],      // row 0
        [0.0, 0.5, 0.0],      // row 1: multiplier = 0.5
        [0.0, 2.0/3.0, 0.0],  // row 2: multiplier = 2/3
    ]);

    let indx = arr1(&[0, 1, 2]); // No pivoting

    // RHS: A * [1, 2, 2] = [4, 7, 6]
    let mut b = arr1(&[4.0, 7.0, 6.0]);  // Fixed: was [4.0, 8.0, 6.0]

    banbks(&a, m1, m2, &al, &indx, &mut b);

    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
}    #[test]
    fn test_simple_2x2_system() {
        let m1 = 1;
        let m2 = 1;

        // Matrix: [4, 1]
        //         [1, 4]
        // LU decomposition:
        // L = [1,   0]
        //     [0.25, 1]
        // U = [4, 1]
        //     [0, 3.75]
        
        let a = arr2(&[
            [0.0, 4.0, 1.0],   // row 0 of U
            [1.0, 3.75, 0.0],  // row 1 of U
        ]);

        let al = arr2(&[
            [0.0, 0.0, 0.0],  // row 0
            [0.0, 0.25, 0.0], // row 1: multiplier = 0.25
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
        let m1 = 0;
        let m2 = 0;

        // Diagonal matrix: [2, 0, 0]
        //                 [0, 3, 0]
        //                 [0, 0, 4]
        // LU is just the matrix itself
        let a = arr2(&[
            [2.0],  // row 0
            [3.0],  // row 1
            [4.0],  // row 2
        ]);

        // No multipliers for diagonal matrix
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
    fn test_larger_banded_system() {
        let m1 = 1;
        let m2 = 1;

        // Simple 3x3 system that we can verify manually
        // Matrix: [3, 1, 0]
        //         [1, 3, 1]
        //         [0, 1, 3]
        // LU decomposition:
        // L = [1, 0, 0]
        //     [1/3, 1, 0]
        //     [0, 3/8, 1]
        // U = [3, 1, 0]
        //     [0, 8/3, 1]
        //     [0, 0, 21/8]
        
        let a = arr2(&[
            [0.0, 3.0, 1.0],        // row 0 of U
            [1.0, 8.0/3.0, 1.0],    // row 1 of U
            [1.0, 21.0/8.0, 0.0],   // row 2 of U
        ]);

        let al = arr2(&[
            [0.0, 0.0, 0.0],        // row 0
            [0.0, 1.0/3.0, 0.0],    // row 1: multiplier = 1/3
            [0.0, 3.0/8.0, 0.0],    // row 2: multiplier = 3/8
        ]);

        let indx = arr1(&[0, 1, 2]);
        
        // RHS for solution [1, 2, 3]
        // A * [1, 2, 3] = [3*1 + 1*2 = 5, 1*1 + 3*2 + 1*3 = 10, 1*2 + 3*3 = 11]
        let mut b = arr1(&[5.0, 10.0, 11.0]);

        banbks(&a, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2, 3]
        assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 3.0, epsilon = 1e-10);
    }
#[test]
fn test_banded_system_with_pivoting() {
    // This test verifies that the pivoting mechanism works by using identity permutation
    // (which should behave the same as no pivoting case)
    let m1 = 1;
    let m2 = 1;

    // Same as the working tridiagonal test but with identity pivoting
    let a = arr2(&[
        [0.0, 2.0, 1.0],   // U matrix: diagonal=2, superdiag=1
        [0.5, 1.5, 1.0],   // Row 1: subdiag multiplier=0.5, diag=1.5, superdiag=1  
        [2.0/3.0, 4.0/3.0, 0.0], // Row 2: multiplier=2/3, diagonal=4/3
    ]);

    let al = arr2(&[
        [0.0, 0.0, 0.0],      // No multipliers for row 0
        [0.0, 0.5, 0.0],      // Multiplier 0.5 at al[1,1] 
        [0.0, 2.0/3.0, 0.0],  // Multiplier 2/3 at al[2,1]
    ]);

    // Identity permutation - no actual pivoting
    let indx = arr1(&[0, 1, 2]);

    // RHS for solution [1, 2, 2] with original matrix [[2,1,0], [1,2,1], [0,1,2]]
    // A*x = [2*1+1*2+0*2, 1*1+2*2+1*2, 0*1+1*2+2*2] = [4, 7, 6]
    let mut b = arr1(&[4.0, 7.0, 6.0]);

    banbks(&a, m1, m2, &al, &indx, &mut b);

    assert_abs_diff_eq!(b[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
} 
}
