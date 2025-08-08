pub fn banbks(
    a: &[Vec<f64>],
    n: usize,
    m1: usize,
    m2: usize,
    a1: &[Vec<f64>],
    indx: &[usize],
    b: &mut [f64],
) {
    let mm = m1 + m2 + 1;
    let mut l = m1;

    // Forward substitution with row swaps
    for k in 1..=n {
        let pivot_row = indx[k];
        if pivot_row != k {
            b.swap(k, pivot_row);
        }

        l = l.min(n);  // Ensure we don't exceed matrix bounds
        if k + m1 < n {
            l += 1;
        }

        // Vectorized elimination step
        for i in (k + 1)..=l {
            b[i] -= a1[k][i - k] * b[k];
        }
    }

    // Back substitution
    l = 1;
    for i in (1..=n).rev() {
        let mut sum = b[i];
        
        // Optimized inner loop
        let row = &a[i];
        for k in 2..=l.min(mm) {
            sum -= row[k] * b[k + i - 1];
        }

        b[i] = sum / a[i][1];
        
        if l < mm {
            l += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn create_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0; cols + 1]; rows + 1] // 1-based indexing
    }

    #[test]
    fn test_tridiagonal_system() {
        let n = 3;
        let m1 = 1;
        let m2 = 1;

        // Example from Numerical Recipes (modified for f64)
        let a = vec![
            vec![],
            vec![0.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 2.0],
        ];

        let a1 = vec![
            vec![],
            vec![0.0, 0.5],      // multipliers from bandec
            vec![0.0, 2.0/3.0],   // (should be 0.666...)
            vec![0.0, 0.0],
        ];

        let indx = vec![0, 1, 2, 3];
        let mut b = vec![0.0, 4.0, 8.0, 6.0];

        banbks(&a, n, m1, m2, &a1, &indx, &mut b);

        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[3], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_banded_system_with_pivoting() {
        let n = 4;
        let m1 = 1;
        let m2 = 2;

        // Example where pivoting would occur
        let a = vec![
            vec![],
            vec![0.0, 0.0, 1.0, 3.0, 0.0],  // Would be pivoted
            vec![0.0, 1.0, 2.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 3.0],
        ];

        // Pretend these are the multipliers from bandec
        let a1 = vec![
            vec![],
            vec![0.0, 1.0, 0.5],  // For row 1
            vec![0.0, 0.5, 0.4],   // For row 2
            vec![0.0, 0.0, 0.3],   // For row 3
            vec![0.0, 0.0, 0.0],
        ];

        // Indicates row 1 and 2 were swapped
        let indx = vec![0, 2, 1, 3, 4];
        let mut b = vec![0.0, 5.0, 6.0, 7.0, 8.0];

        banbks(&a, n, m1, m2, &a1, &indx, &mut b);

        // Verify solution (values based on expected results)
        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[3], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[4], 2.0, epsilon = 1e-10);
    }
}
