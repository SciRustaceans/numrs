pub fn banbks(
    a: &[Vec<f64>],
    n: usize,
    m1: usize,
    m2: usize,
    al: &[Vec<f64>],  // Changed from a1 to al for clarity (lower triangular multipliers)
    indx: &[usize],
    b: &mut [f64],
) {
    let mm = m1 + m2 + 1;
    let mut l = m1;

    // Forward substitution
    for k in 1..=n {
        let i = indx[k];
        if i != k {
            b.swap(i, k);
        }
        if k < n {
            l = l.min(n - k);
            for j in 1..=l {
                b[k + j] -= al[k][j] * b[k];
            }
        }
    }

    // Back substitution
    l = 1;
    for i in (1..=n).rev() {
        let mut sum = b[i];
        if i < n {
            let k_max = l.min(n - i);
            for k in 1..=k_max {
                sum -= a[i][m1 + 1 + k] * b[i + k];
            }
        }
        b[i] = sum / a[i][m1 + 1];
        if l < mm {
            l += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tridiagonal_system() {
        let n = 3;
        let m1 = 1;  // lower bandwidth
        let m2 = 1;  // upper bandwidth

        // Band matrix stored in compact form (1-based indexing)
        // Matrix: [2, 1, 0]
        //         [1, 2, 1] 
        //         [0, 1, 2]
        // Stored as:
        // row 1: [*, 2, 1]   (* = unused)
        // row 2: [1, 2, 1]
        // row 3: [1, 2, *]
        let a = vec![
            vec![],           // row 0 (unused)
            vec![0.0, 0.0, 2.0, 1.0],  // row 1: [*, 2, 1]
            vec![0.0, 1.0, 2.0, 1.0],  // row 2: [1, 2, 1]  
            vec![0.0, 1.0, 2.0, 0.0],  // row 3: [1, 2, *]
        ];

        // Lower triangular multipliers from bandec
        // These would come from the bandec decomposition
        let al = vec![
            vec![],           // row 0 (unused)
            vec![0.0, 0.0, 0.0],  // row 1 multipliers
            vec![0.0, 0.5, 0.0],  // row 2: multiplier = 0.5
            vec![0.0, 2.0/3.0, 0.0], // row 3: multiplier = 2/3
        ];

        // Pivot indices from bandec (1-based)
        let indx = vec![0, 1, 2, 3];  // No pivoting in this case

        // Right-hand side: A * [1, 2, 2] = [4, 8, 6]
        let mut b = vec![0.0, 4.0, 8.0, 6.0];

        banbks(&a, n, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2, 2]
        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[3], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simple_2x2_system() {
        let n = 2;
        let m1 = 1;
        let m2 = 1;

        // Matrix: [4, 1]
        //         [1, 4]
        let a = vec![
            vec![],
            vec![0.0, 0.0, 4.0, 1.0],  // row 1: [*, 4, 1]
            vec![0.0, 1.0, 4.0, 0.0],  // row 2: [1, 4, *]
        ];

        // Multipliers from bandec
        let al = vec![
            vec![],
            vec![0.0, 0.0, 0.0],      // row 1
            vec![0.0, 0.25, 0.0],     // row 2: multiplier = 1/4
        ];

        let indx = vec![0, 1, 2];
        // RHS: [4*1 + 1*2 = 6, 1*1 + 4*2 = 9]
        let mut b = vec![0.0, 6.0, 9.0];

        banbks(&a, n, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2]
        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_system() {
        let n = 3;
        let m1 = 0;  // no lower diagonal
        let m2 = 0;  // no upper diagonal

        // Diagonal matrix: [2, 0, 0]
        //                 [0, 3, 0]
        //                 [0, 0, 4]
        let a = vec![
            vec![],
            vec![0.0, 2.0],  // row 1: [2]
            vec![0.0, 3.0],  // row 2: [3]
            vec![0.0, 4.0],  // row 3: [4]
        ];

        // No multipliers for diagonal matrix
        let al = vec![
            vec![],
            vec![0.0],
            vec![0.0],
            vec![0.0],
        ];

        let indx = vec![0, 1, 2, 3];
        // RHS: [2*1=2, 3*2=6, 4*3=12]
        let mut b = vec![0.0, 2.0, 6.0, 12.0];

        banbks(&a, n, m1, m2, &al, &indx, &mut b);

        // Solution should be [1, 2, 3]
        assert_abs_diff_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b[3], 3.0, epsilon = 1e-10);
    }
}
