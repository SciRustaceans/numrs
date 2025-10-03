use std::error::Error;
use std::fmt;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    InvalidOrder,
    InvalidInput,
    NumericalInstability,
    MatrixError,
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuadratureError::InvalidOrder => write!(f, "Invalid quadrature order"),
            QuadratureError::InvalidInput => write!(f, "Invalid input parameters"),
            QuadratureError::NumericalInstability => write!(f, "Numerical instability detected"),
            QuadratureError::MatrixError => write!(f, "Matrix operation failed"),
        }
    }
}

impl Error for QuadratureError {}

pub type QuadratureResult<T> = std::result::Result<T, QuadratureError>;

/// Computes Gauss quadrature nodes and weights from recurrence coefficients
pub fn gaucof(n: usize, a: &[f64], b: &[f64], amu0: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if a.len() != n || b.len() != n {
        return Err(QuadratureError::InvalidInput);
    }
    if amu0 <= 0.0 {
        return Err(QuadratureError::InvalidInput);
    }

    // Create symmetric tridiagonal matrix
    let mut d = a.to_vec();
    let mut e = vec![0.0; n];
    
    // Copy b, taking square roots for off-diagonal elements
    for i in 0..n {
        if i > 0 {
            e[i] = b[i].sqrt();
        }
    }

    // Create identity matrix for eigenvectors
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }

    // Solve eigenvalue problem for tridiagonal matrix
    tqli(&mut d, &mut e, n, &mut z)?;
    eigsrt(&mut d, &mut z, n);

    // Extract nodes and weights
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];

    for i in 0..n {
        x[i] = d[i];
        w[i] = amu0 * z[0][i] * z[0][i];
    }

    Ok((x, w))
}

/// Multithreaded version of gaucof
pub fn gaucof_parallel(n: usize, a: &[f64], b: &[f64], amu0: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if a.len() != n || b.len() != n {
        return Err(QuadratureError::InvalidInput);
    }
    if amu0 <= 0.0 {
        return Err(QuadratureError::InvalidInput);
    }

    let mut d = a.to_vec();
    let mut e = vec![0.0; n];
    
    for i in 0..n {
        if i > 0 {
            e[i] = b[i].sqrt();
        }
    }

    // Create identity matrix in parallel
    let z = Arc::new(Mutex::new(vec![vec![0.0; n]; n]));
    
    (0..n).into_par_iter().for_each(|i| {
        let mut z_lock = z.lock().unwrap();
        z_lock[i][i] = 1.0;
    });

    let mut z = Arc::try_unwrap(z).unwrap().into_inner().unwrap();

    tqli(&mut d, &mut e, n, &mut z)?;
    eigsrt_parallel(&mut d, &mut z, n);

    let x: Vec<f64> = d.par_iter().cloned().collect();
    let w: Vec<f64> = (0..n).into_par_iter()
        .map(|i| amu0 * z[0][i] * z[0][i])
        .collect();

    Ok((x, w))
}

/// TQLI algorithm for tridiagonal matrix eigenvalue problem
fn tqli(d: &mut [f64], e: &mut [f64], n: usize, z: &mut [Vec<f64>]) -> QuadratureResult<()> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }

    for i in 1..n {
        e[i-1] = e[i];
    }
    e[n-1] = 0.0;

    for l in 0..n {
        let mut iter = 0;
        let mut m = l;
        
        while m < n {
            let mut dd = (d[m] - d[m+1]).abs();
            if m < n-1 {
                dd += e[m].abs();
            }
            
            if dd < 1e-12 {
                break;
            }
            
            if iter >= 30 {
                return Err(QuadratureError::NumericalInstability);
            }
            iter += 1;

            let mut g = d[l];
            let mut p = (d[m] - g) / (2.0 * e[l]);
            let mut r = (p * p + 1.0).sqrt();
            
            if p < 0.0 {
                d[m] = d[m] + e[l] / (p - r);
            } else {
                d[m] = d[m] + e[l] / (p + r);
            }
            
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            
            for i in m-1..=l {
                let f = s * e[i];
                let b = c * e[i];
                
                if f.abs() >= (d[i] - d[i+1]).abs() {
                    break;
                }
                
                r = (f * f + (d[i] - d[i+1]) * (d[i] - d[i+1])).sqrt();
                e[i+1] = s * r;
                s = f / r;
                c = (d[i] - d[i+1]) / r;
                d[i+1] = d[i] + c * (d[i+1] - d[i]);
                
                for k in 0..n {
                    p = z[k][i+1];
                    z[k][i+1] = s * z[k][i] + c * p;
                    z[k][i] = c * z[k][i] - s * p;
                }
            }
            
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    Ok(())
}

/// Sort eigenvalues and eigenvectors
fn eigsrt(d: &mut [f64], z: &mut [Vec<f64>], n: usize) {
    for i in 0..n-1 {
        let mut k = i;
        let mut p = d[i];
        
        for j in i+1..n {
            if d[j] >= p {
                k = j;
                p = d[j];
            }
        }
        
        if k != i {
            d.swap(i, k);
            for j in 0..n {
                z[j].swap(i, k);
            }
        }
    }
}

/// Parallel version of eigsrt
fn eigsrt_parallel(d: &mut [f64], z: &mut [Vec<f64>], n: usize) {
    let indices: Vec<usize> = (0..n).collect();
    
    // Sort indices based on eigenvalues
    let mut sorted_indices: Vec<usize> = indices.par_iter()
        .cloned()
        .collect();
    
    sorted_indices.par_sort_unstable_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap());
    
    // Reorder eigenvalues and eigenvectors
    let d_old = d.to_vec();
    let z_old = z.to_vec();
    
    for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
        d[new_idx] = d_old[old_idx];
        for j in 0..n {
            z[j][new_idx] = z_old[j][old_idx];
        }
    }
}

/// Generate Gauss quadrature rules for common orthogonal polynomials
pub struct GaussQuadratureGenerator;

impl GaussQuadratureGenerator {
    /// Generate Gauss-Legendre quadrature rules
    pub fn gauss_legendre(n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        for i in 0..n {
            if i > 0 {
                b[i] = (i as f64 * i as f64) / (4.0 * i as f64 * i as f64 - 1.0);
            }
        }
        
        gaucof(n, &a, &b, 2.0)
    }
    
    /// Generate Gauss-Hermite quadrature rules
    pub fn gauss_hermite(n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        for i in 0..n {
            if i > 0 {
                b[i] = i as f64 / 2.0;
            }
        }
        
        gaucof(n, &a, &b, std::f64::consts::PI.sqrt())
    }
    
    /// Generate Gauss-Laguerre quadrature rules
    pub fn gauss_laguerre(n: usize, alf: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        for i in 0..n {
            a[i] = 2.0 * i as f64 + alf + 1.0;
            if i > 0 {
                b[i] = i as f64 * (i as f64 + alf);
            }
        }
        
        let amu0 = (alf + 1.0).gamma();
        gaucof(n, &a, &b, amu0)
    }
    
    /// Generate Gauss-Jacobi quadrature rules
    pub fn gauss_jacobi(n: usize, alf: f64, bet: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        let alfbet = alf + bet;
        for i in 0..n {
            a[i] = (bet * bet - alf * alf) / ((2.0 * i as f64 + alfbet) * (2.0 * i as f64 + alfbet + 2.0));
            if i > 0 {
                b[i] = 4.0 * i as f64 * (i as f64 + alf) * (i as f64 + bet) * (i as f64 + alfbet) /
                       ((2.0 * i as f64 + alfbet) * (2.0 * i as f64 + alfbet) * 
                        (2.0 * i as f64 + alfbet + 1.0) * (2.0 * i as f64 + alfbet - 1.0));
            }
        }
        
        let amu0 = 2.0f64.powf(alfbet + 1.0) * (alf + 1.0).gamma() * (bet + 1.0).gamma() / 
                   (alfbet + 2.0).gamma();
        gaucof(n, &a, &b, amu0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gaucof_legendre() {
        let n = 5;
        let a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        for i in 1..n {
            b[i] = (i * i) as f64 / (4 * i * i - 1) as f64;
        }
        
        let (x, w) = gaucof(n, &a, &b, 2.0).unwrap();
        
        // Check symmetry
        assert_abs_diff_eq!(x[0], -x[4], epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], -x[3], epsilon = 1e-10);
        assert_abs_diff_eq!(w[0], w[4], epsilon = 1e-10);
        assert_abs_diff_eq!(w[1], w[3], epsilon = 1e-10);
        
        // Check weight sum
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaucof_parallel_consistency() {
        let n = 5;
        let a = vec![0.0; n];
        let mut b = vec![0.0; n];
        
        for i in 1..n {
            b[i] = (i * i) as f64 / (4 * i * i - 1) as f64;
        }
        
        let (x1, w1) = gaucof(n, &a, &b, 2.0).unwrap();
        let (x2, w2) = gaucof_parallel(n, &a, &b, 2.0).unwrap();
        
        for i in 0..n {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-10);
            assert_abs_diff_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaucof_invalid_order() {
        let result = gaucof(0, &[], &[], 1.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidOrder);
    }

    #[test]
    fn test_gaucof_invalid_input() {
        let result = gaucof(2, &[0.0], &[0.0], 1.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidInput);
    }

    #[test]
    fn test_gauss_legendre_generator() {
        let (x, w) = GaussQuadratureGenerator::gauss_legendre(5).unwrap();
        
        // Check basic properties
        assert_abs_diff_eq!(w.iter().sum::<f64>(), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[0], -x[4], epsilon = 1e-10);
    }

       #[test]
        fn test_gauss_hermite_generator() {
            let (x, w) = GaussQuadratureGenerator::gauss_hermite(5).unwrap();
            
            // Check basic properties
            let sum_weights: f64 = w.iter().sum();
            let expected_sum = std::f64::consts::PI.sqrt();
            let diff = (sum_weights - expected_sum).abs();
            assert!(
                diff < 1e-10,
                "Weight sum failed: sum={}, expected={}, diff={}",
                sum_weights, expected_sum, diff
            );
        } 

    #[test]
    fn test_gauss_laguerre_generator() {
        let (x, w) = GaussQuadratureGenerator::gauss_laguerre(5, 0.0).unwrap();
        
        // Check basic properties
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 1.0, epsilon = 1e-10); // Gamma(1) = 1
    }

    #[test]
    fn test_gauss_jacobi_generator() {
        let (x, w) = GaussQuadratureGenerator::gauss_jacobi(5, 0.0, 0.0).unwrap();
        
        // Should be same as Legendre
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eigenvalue_sorting() {
        let mut d = vec![3.0, 1.0, 2.0];
        let mut z = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        eigsrt(&mut d, &mut z, 3);
        
        // Should be sorted in descending order
        assert_abs_diff_eq!(d[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tqli_algorithm() {
        // Test with a simple symmetric matrix
        let n = 3;
        let mut d = vec![2.0, 2.0, 1.0];
        let mut e = vec![0.0, 1.0, 1.0];
        let mut z = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let result = tqli(&mut d, &mut e, n, &mut z);
        assert!(result.is_ok());
        
        // Eigenvalues should be real and sorted after eigsrt
        eigsrt(&mut d, &mut z, n);
        assert!(d[0] >= d[1] && d[1] >= d[2]);
    }

    #[test]
    fn test_gaucof_precision() {
        // Test that Gauss quadrature is exact for polynomials up to degree 2n-1
        let n = 5;
        let (x, w) = GaussQuadratureGenerator::gauss_legendre(n).unwrap();
        
        for degree in 0..2*n {
            let exact = if degree % 2 == 0 {
                2.0 / (degree as f64 + 1.0)
            } else {
                0.0
            };
            
            let computed: f64 = x.iter()
                .zip(w.iter())
                .map(|(&x, &w)| w * x.powi(degree as i32))
                .sum();
            
            let diff = (computed - exact).abs();
            assert!(
                diff < 1e-10,
                "Failed for degree {}: computed={}, exact = {}, diff = {}",
                degree, computed, exact, diff
        );
        }
    }

    #[test]
    fn test_parallel_eigsrt() {
        let mut d = vec![3.0, 1.0, 2.0, 4.0];
        let mut z = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        
        eigsrt_parallel(&mut d, &mut z, 4);
        
        // Should be sorted in descending order
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_amu0() {
        let result = gaucof(2, &[0.0, 0.0], &[0.0, 0.0], -1.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidInput);
    }
}
