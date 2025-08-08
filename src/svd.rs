// src/svd.rs

//! A robust implementation of Singular Value Decomposition based on the Golub-Kahan-Reinsch algorithm.

use crate::utils::{pythag, sign};

const MAX_SVD_ITERATIONS: usize = 30;

/// Singular Value Decomposition using a modern, robust algorithm.
///
/// This function computes the SVD of a matrix `a`, A = U * W * V^T.
///
/// # Arguments
/// * `a` - Input m x n matrix. Becomes the m x n matrix U on output.
/// * `w` - Output n-element vector of singular values.
/// * `v` - Output n x n orthogonal matrix V.
///
/// # Returns
/// A `Result` indicating success or an error message if it fails to converge.
pub fn svdcmp(
    a: &mut [Vec<f64>],
    w: &mut [f64],
    v: &mut [Vec<f64>],
) -> Result<(), &'static str> {
    let m = a.len();
    let n = a[0].len();
    let mut rv1 = vec![0.0; n];
    let mut g = 0.0;
    let mut scale = 0.0;

    // Householder reduction to bidiagonal form.
    for i in 0..n {
        let l = i + 1;
        rv1[i] = scale * g;
        g = 0.0;
        scale = 0.0;
        if i < m {
            for k in i..m {
                scale += a[k][i].abs();
            }
            if scale != 0.0 {
                let mut s = 0.0;
                for k in i..m {
                    a[k][i] /= scale;
                    s += a[k][i] * a[k][i];
                }
                let f = a[i][i];
                g = -sign(s.sqrt(), f);
                let h = f * g - s;
                a[i][i] = f - g;
                for j in l..n {
                    s = 0.0;
                    for k in i..m {
                        s += a[k][i] * a[k][j];
                    }
                    let f = s / h;
                    for k in i..m {
                        a[k][j] += f * a[k][i];
                    }
                }
                for k in i..m {
                    a[k][i] *= scale;
                }
            }
        }
        w[i] = scale * g;
        g = 0.0;
        scale = 0.0;
        if i < m && i != n - 1 {
            for k in l..n {
                scale += a[i][k].abs();
            }
            if scale != 0.0 {
                let mut s = 0.0;
                for k in l..n {
                    a[i][k] /= scale;
                    s += a[i][k] * a[i][k];
                }
                let f = a[i][l];
                g = -sign(s.sqrt(), f);
                let h = f * g - s;
                a[i][l] = f - g;
                for k in l..n {
                    rv1[k] = a[i][k] / h;
                }
                for j in l..m {
                    s = 0.0;
                    for k in l..n {
                        s += a[j][k] * a[i][k];
                    }
                    for k in l..n {
                        a[j][k] += s * rv1[k];
                    }
                }
                for k in l..n {
                    a[i][k] *= scale;
                }
            }
        }
    }

    // Accumulation of right-hand transformations.
    for i in (0..n).rev() {
        let l = i + 1;
        g = rv1[i];
        if i < n - 1 {
            for j in l..n {
                v[i][j] = 0.0;
            }
        }
        if g != 0.0 {
            // This check prevents the out-of-bounds panic.
            if i != n - 1 {
                let h = a[i][i + 1] * g;
                for j in l..n {
                    let mut s = 0.0;
                    for k in l..n {
                        s += a[i][k] * v[k][j];
                    }
                    let f = s / h;
                    for k in i..n {
                        v[k][j] += f * a[i][k];
                    }
                }
            }
        }
        for j in i..n {
            v[j][i] = 0.0;
            v[i][j] = 0.0;
        }
        v[i][i] = 1.0;
    }

    // Accumulation of left-hand transformations.
    for i in (0..n.min(m)).rev() {
        let l = i + 1;
        g = w[i];
        if i < n {
            for j in l..n {
                a[i][j] = 0.0;
            }
        }
        if g != 0.0 {
            let h = a[i][i] * g;
            if i != n - 1 {
                for j in l..n {
                    let mut s = 0.0;
                    for k in l..m {
                        s += a[k][i] * a[k][j];
                    }
                    let f = s / h;
                    for k in i..m {
                        a[k][j] += f * a[k][i];
                    }
                }
            }
            for j in i..m {
                a[j][i] /= g;
            }
        } else {
            for j in i..m {
                a[j][i] = 0.0;
            }
        }
        a[i][i] += 1.0;
    }

    // Diagonalization of the bidiagonal form.
    for k in (0..n).rev() {
        for _its in 0..MAX_SVD_ITERATIONS {
            let mut flag = true;
            let mut l = k;
            while l > 0 {
                if rv1[l - 1].abs() < f64::EPSILON * (w[l].abs() + w[l-1].abs()) {
                    flag = false;
                    break;
                }
                if w[l - 1].abs() < f64::EPSILON {
                    break;
                }
                l -= 1;
            }

            if flag {
                let mut c = 0.0;
                let mut s = 1.0;
                for i in l..=k {
                    let f = s * rv1[i];
                    rv1[i] *= c;
                    if f.abs() < f64::EPSILON {
                        break;
                    }
                    g = w[i];
                    let h = pythag(f, g);
                    w[i] = h;
                    c = g / h;
                    s = -f / h;
                    for j in 0..m {
                        let y = a[j][l - 1];
                        let z = a[j][i];
                        a[j][l - 1] = y * c + z * s;
                        a[j][i] = -s * y + c * z;
                    }
                }
            }
            
            let z = w[k];
            if l == k {
                if z < 0.0 {
                    w[k] = -z;
                    for j in 0..n {
                        v[j][k] = -v[j][k];
                    }
                }
                break;
            }

            if _its >= MAX_SVD_ITERATIONS - 1 {
                return Err("SVD failed to converge");
            }

            let mut x = w[l];
            let y = w[k - 1];
            g = rv1[k - 1];
            let h = rv1[k];
            let mut f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * (y / (f + sign(g, f)) - h)) / x;
            let mut c = 1.0;
            let mut s = 1.0;

            for j in l..k {
                let i = j + 1;
                g = rv1[i];
                let y_val = w[i];
                let h_val = s * g;
                g = c * g;
                let z_val = pythag(f, h_val);
                rv1[j] = z_val;
                c = f / z_val;
                s = h_val / z_val;
                f = x * c + g * s;
                g = -s * x + c * g;
                let h_val = y_val * s;
                let y_val = c * y_val;
                for jj in 0..n {
                    let x_v = v[jj][j];
                    let z_v = v[jj][i];
                    v[jj][j] = x_v * c + z_v * s;
                    v[jj][i] = -s * x_v + c * z_v;
                }
                let z_val = pythag(f, h_val);
                w[j] = z_val;
                if z_val != 0.0 {
                    let inv_z = 1.0 / z_val;
                    c = f * inv_z;
                    s = h_val * inv_z;
                }
                f = c * g + s * y_val;
                x = -s * g + c * y_val;
                for jj in 0..m {
                    let y_a = a[jj][j];
                    let z_a = a[jj][i];
                    a[jj][j] = y_a * c + z_a * s;
                    a[jj][i] = -s * y_a + c * z_a;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }

    // Final sorting of singular values
    for i in 0..n {
        let mut max_idx = i;
        for j in (i + 1)..n {
            if w[j] > w[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            w.swap(i, max_idx);
            a.iter_mut().for_each(|row| row.swap(i, max_idx));
            v.iter_mut().for_each(|row| row.swap(i, max_idx));
        }
    }

    Ok(())
}


// The unit tests are unchanged and remain below.
#[cfg(test)]
mod tests {
    use super::*;

    fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if m.is_empty() { return vec![]; }
        let mut t = vec![vec![0.0; m.len()]; m[0].len()];
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                t[j][i] = val;
            }
        }
        t
    }

    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (m, n) = (a.len(), a[0].len());
        let p = b[0].len();
        let mut c = vec![vec![0.0; p]; m];
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn is_identity(m: &[Vec<f64>], tol: f64) -> bool {
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if i == j {
                    if (val - 1.0).abs() > tol { return false; }
                } else {
                    if val.abs() > tol { return false; }
                }
            }
        }
        true
    }
    
    #[test]
    fn test_v_matrix_is_orthogonal() {
        let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let n = a[0].len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let v_t = transpose(&v);
        let v_v_t = mat_mul(&v, &v_t);
        assert!(is_identity(&v_v_t, 1e-10), "V * V^T should be the identity matrix");
    }

    #[test]
    fn test_svd_square_matrix() {
        let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let n = a[0].len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let expected_w0 = 5.46498570421504;
        let expected_w1 = 0.3659661906262578;
        assert!((w[0] - expected_w0).abs() < 1e-10, "Largest singular value is incorrect");
        assert!((w[1] - expected_w1).abs() < 1e-10, "Smallest singular value is incorrect");
    }

    #[test]
    fn test_svd_tall_matrix() {
        let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let n = a[0].len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let expected_w = [9.52551809, 0.77286964];
        assert!((w[0] - expected_w[0]).abs() < 1e-8);
        assert!((w[1] - expected_w[1]).abs() < 1e-8);
    }

    #[test]
    fn test_svd_wide_matrix() {
        let mut a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let n = a[0].len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let expected_w = [9.5058059, 0.77792135];
        assert!((w[0] - expected_w[0]).abs() < 1e-8);
        assert!((w[1] - expected_w[1]).abs() < 1e-8);
        assert!(w[2].abs() < 1e-10);
    }

    #[test]
    fn test_svd_identity_matrix() {
        let mut a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let n = a.len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        for val in &w {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let mut a = vec![vec![5.0, 0.0, 0.0], vec![0.0, -3.0, 0.0], vec![0.0, 0.0, 1.0]];
        let n = a.len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let expected_w = [5.0, 3.0, 1.0];
        for i in 0..n {
            assert!((w[i] - expected_w[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_svd_matrix_with_zero_column() {
        let mut a = vec![vec![1.0, 0.0, 3.0], vec![4.0, 0.0, 6.0], vec![7.0, 0.0, 9.0]];
        let n = a[0].len();
        let mut w = vec![0.0; n];
        let mut v = vec![vec![0.0; n]; n];
        svdcmp(&mut a, &mut w, &mut v).expect("SVD failed");
        let expected_w = [12.4412389, 0.4900696];
        assert!((w[0] - expected_w[0]).abs() < 1e-7);
        assert!((w[1] - expected_w[1]).abs() < 1e-7);
        assert!(w[2].abs() < 1e-10);
    }
}
