use super::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;

#[rstest]
#[case(3, 1, 1)]  // Tridiagonal
#[case(5, 2, 1)]  // Pentadiagonal
fn test_banded_lu_decomposition(
    #[case] n: usize,
    #[case] m1: usize,
    #[case] m2: usize,
) {
    // Create diagonally dominant band matrix
    let mut a = create_band_matrix(n, m1, m2);
    for i in 1..=n {
        for j in 1..=(m1 + m2 + 1) {
            let diag = if j == m1 + 1 { 4.0 } else { 1.0 };
            a[i][j] = diag / (i + j) as f64;
        }
    }

    let mut al = vec![vec![0.0; m1 + 1]; n + 1];
    let mut indx = vec![0; n + 1];
    let mut d = 0.0;

    bandec(&mut a, n, m1, m2, &mut al, &mut indx, &mut d);

    // Verify solution for multiple RHS vectors
    for trial in 1..=3 {
        let mut b = vec![0.0; n + 1];
        let mut x_true = vec![0.0; n + 1];
        for i in 1..=n {
            b[i] = (i * trial) as f64;
            x_true[i] = i as f64;
        }

        banbks(&a, n, m1, m2, &al, &indx, &mut b);

        // Verify solution matches within tolerance
        for i in 1..=n {
            assert_abs_diff_eq!(b[i], x_true[i], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_singular_system_handling() {
    let n = 3;
    let m1 = 1;
    let m2 = 1;

    // Create singular matrix
    let mut a = create_band_matrix(n, m1, m2);
    for i in 1..=n {
        a[i][1] = 1.0;
        a[i][2] = 1.0;
    }

    let mut al = vec![vec![0.0; m1 + 1]; n + 1];
    let mut indx = vec![0; n + 1];
    let mut d = 0.0;

    bandec(&mut a, n, m1, m2, &mut al, &mut indx, &mut d);

    // Should handle singular system with TINY perturbation
    assert!(a[1][1].abs() > 0.0 || a[1][1] == TINY);
}
