use super::*;
use approx::assert_abs_diff_eq;
use rand::Rng;

fn create_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; cols + 1]; rows + 1] // 1-based indexing
}

#[test]
fn test_svd_identity_matrix() {
    let n = 3;
    let mut a = create_matrix(n, n);
    for i in 1..=n {
        a[i][i] = 1.0;
    }
    
    let mut w = vec![0.0; n + 1];
    let mut v = create_matrix(n, n);
    
    svdcmp(&mut a, &mut w, &mut v).unwrap();
    
    // All singular values should be 1
    for i in 1..=n {
        assert_abs_diff_eq!(w[i], 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_svd_diagonal_matrix() {
    let n = 4;
    let mut a = create_matrix(n, n);
    let test_values = [2.0, 4.0, 1.0, 3.0];
    for i in 1..=n {
        a[i][i] = test_values[i-1];
    }
    
    let mut w = vec![0.0; n + 1];
    let mut v = create_matrix(n, n);
    
    svdcmp(&mut a, &mut w, &mut v).unwrap();
    
    // Singular values should match diagonal elements in descending order
    let mut sorted_values = test_values;
    sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for i in 1..=n {
        assert_abs_diff_eq!(w[i], sorted_values[i-1], epsilon = 1e-10);
    }
}

#[test]
fn test_svd_random_matrix() {
    let n = 5;
    let mut rng = rand::thread_rng();
    let mut a = create_matrix(n, n);
    for i in 1..=n {
        for j in 1..=n {
            a[i][j] = rng.gen_range(-1.0..1.0);
        }
    }
    
    let mut w = vec![0.0; n + 1];
    let mut v = create_matrix(n, n);
    
    svdcmp(&mut a, &mut w, &mut v).unwrap();
    
    // Verify singular values are non-negative and sorted
    for i in 1..n {
        assert!(w[i] >= w[i+1]);
    }
    assert!(w[n] >= 0.0);
    
    // Verify reconstruction A = U Σ V^T
    let mut reconstructed = create_matrix(n, n);
    for i in 1..=n {
        for j in 1..=n {
            for k in 1..=n {
                reconstructed[i][j] += a[i][k] * w[k] * v[j][k];
            }
        }
    }
    
    for i in 1..=n {
        for j in 1..=n {
            assert_abs_diff_eq!(reconstructed[i][j], a[i][j], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_svd_orthogonality() {
    let n = 4;
    let mut a = create_matrix(n, n);
    let mut rng = rand::thread_rng();
    for i in 1..=n {
        for j in 1..=n {
            a[i][j] = rng.gen_range(-1.0..1.0);
        }
    }
    
    let mut w = vec![0.0; n + 1];
    let mut v = create_matrix(n, n);
    
    svdcmp(&mut a, &mut w, &mut v).unwrap();
    
    // Verify V is orthogonal (V^T V = I)
    for i in 1..=n {
        for j in 1..=n {
            let dot: f64 = (1..=n).map(|k| v[k][i] * v[k][j]).sum();
            if i == j {
                assert_abs_diff_eq!(dot, 1.0, epsilon = 1e-10);
            } else {
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_svd_convergence() {
    let n = 10;
    let mut a = create_matrix(n, n);
    // Create Hilbert matrix (known to be ill-conditioned)
    for i in 1..=n {
        for j in 1..=n {
            a[i][j] = 1.0 / ((i + j - 1) as f64);
        }
    }
    
    let mut w = vec![0.0; n + 1];
    let mut v = create_matrix(n, n);
    
    assert!(svdcmp(&mut a, &mut w, &mut v).is_ok());
    
    // Verify singular values are positive and ordered
    for i in 1..n {
        assert!(w[i] >= w[i+1]);
    }
    assert!(w[n] > 0.0);
}

#[test]
fn test_svd_error_handling() {
    // Empty matrix
    let mut a: Vec<Vec<f64>> = vec![vec![]];
    let mut w = vec![];
    let mut v: Vec<Vec<f64>> = vec![vec![]];
    
    assert!(svdcmp(&mut a, &mut w, &mut v).is_err());
    
    // Mismatched dimensions
    let mut a = vec![vec![0.0; 3]; 2]; // 2x3 matrix (0-based)
    let mut w = vec![0.0; 3];
    let mut v = vec![vec![0.0; 3]; 3];
    
    // Need to convert to 1-based indexing
    a.insert(0, vec![]);
    for row in &mut a {
        row.insert(0, 0.0);
    }
    w.insert(0, 0.0);
    v.insert(0, vec![]);
    for row in &mut v {
        row.insert(0, 0.0);
    }
    
    assert!(svdcmp(&mut a, &mut w, &mut v).is_ok());
}
