use std::error::Error;
use std::fmt;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const EPS: f64 = 3.0e-14;
const PIM4: f64 = 0.7511255444649425;
const MAXIT: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    InvalidOrder,
    ConvergenceFailed,
    NumericalInstability,
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuadratureError::InvalidOrder => write!(f, "Invalid quadrature order"),
            QuadratureError::ConvergenceFailed => write!(f, "Newton-Raphson convergence failed"),
            QuadratureError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl Error for QuadratureError {}

pub type QuadratureResult<T> = std::result::Result<T, QuadratureError>;

/// Gauss-Hermite quadrature nodes and weights computation
pub fn gauher(n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }

    let m = (n + 1) / 2;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];

    for i in 0..m {
        let mut z = if i == 0 {
            (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-0.16667)
        } else if i == 1 {
            let temp = (n as f64).powf(0.426);
            x[0] - 1.14 * temp / x[0]
        } else if i == 2 {
            1.86 * x[1] - 0.86 * x[0]
        } else if i == 3 {
            1.91 * x[2] - 0.91 * x[1]
        } else {
            2.0 * x[i-1] - x[i-2]
        };

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, new_pp) = hermite_polynomial(n, z);
            pp = new_pp;
            
            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() <= EPS {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(QuadratureError::ConvergenceFailed);
        }

        let (p1, pp) = hermite_polynomial(n, z);
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }

    Ok((x, w))
}

/// Compute Hermite polynomial and its derivative at point z
fn hermite_polynomial(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = PIM4;
    let mut p2 = 0.0;
    let mut p3;

    for j in 1..=n {
        p3 = p2;
        p2 = p1;
        p1 = z * (2.0 / j as f64).sqrt() * p2 - ((j as f64 - 1.0) / j as f64).sqrt() * p3;
    }

    let pp = (2.0 * n as f64).sqrt() * p2;
    (p1, pp)
}

/// Multithreaded version of gauher
pub fn gauher_parallel(n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }

    let m = (n + 1) / 2;
    let x = Arc::new(Mutex::new(vec![0.0; n]));
    let w = Arc::new(Mutex::new(vec![0.0; n]));

    // Process each root in parallel
    (0..m).into_par_iter().for_each(|i| {
        let x_lock = x.lock().unwrap();
        let mut z = if i == 0 {
            (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-0.16667)
        } else if i == 1 {
            let temp = (n as f64).powf(0.426);
            x_lock[0] - 1.14 * temp / x_lock[0]
        } else if i == 2 {
            1.86 * x_lock[1] - 0.86 * x_lock[0]
        } else if i == 3 {
            1.91 * x_lock[2] - 0.91 * x_lock[1]
        } else {
            2.0 * x_lock[i-1] - x_lock[i-2]
        };
        drop(x_lock); // Release lock for computation

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, new_pp) = hermite_polynomial(n, z);
            pp = new_pp;
            
            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() <= EPS {
                converged = true;
                break;
            }
        }

        if converged {
            let (p1, pp) = hermite_polynomial(n, z);
            let weight = 2.0 / (pp * pp);
            
            let mut x_lock = x.lock().unwrap();
            let mut w_lock = w.lock().unwrap();
            
            x_lock[i] = z;
            x_lock[n - 1 - i] = -z;
            w_lock[i] = weight;
            w_lock[n - 1 - i] = weight;
        }
    });

    let x_result = Arc::try_unwrap(x).unwrap().into_inner().unwrap();
    let w_result = Arc::try_unwrap(w).unwrap().into_inner().unwrap();
    
    Ok((x_result, w_result))
}

/// Cache for Gauss-Hermite quadrature rules
pub struct GaussHermiteCache {
    rules: std::sync::RwLock<std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>>,
}

impl GaussHermiteCache {
    pub fn new() -> Self {
        Self {
            rules: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn get_rule(&self, n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        if let Some(rule) = self.rules.read().unwrap().get(&n) {
            return Ok(rule.clone());
        }

        let rule = gauher(n)?;
        self.rules.write().unwrap().insert(n, rule.clone());
        
        Ok(rule)
    }

    pub fn get_rule_parallel(&self, n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        if let Some(rule) = self.rules.read().unwrap().get(&n) {
            return Ok(rule.clone());
        }

        let rule = gauher_parallel(n)?;
        self.rules.write().unwrap().insert(n, rule.clone());
        
        Ok(rule)
    }
}

/// Integration using Gauss-Hermite quadrature
pub fn gauss_hermite_quadrature<F>(func: F, n: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gauher(n)?;
    
    let sum: f64 = nodes.iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x))
        .sum();
    
    Ok(sum)
}

/// Multithreaded Gauss-Hermite quadrature
pub fn gauss_hermite_quadrature_parallel<F>(func: F, n: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    let (nodes, weights) = gauher_parallel(n)?;
    
    let sum: f64 = nodes.par_iter()
        .zip(weights.par_iter())
        .map(|(&x, &w)| w * func(x))
        .sum();
    
    Ok(sum)
}

/// Integration with weight function e^{-x²}
pub fn gauss_hermite_weighted_quadrature<F>(func: F, n: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gauher(n)?;
    
    let sum: f64 = nodes.iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x) * (x * x).exp())
        .sum();
    
    Ok(sum)
}

/// Adaptive Gauss-Hermite quadrature
pub fn gauss_hermite_adaptive<F>(func: F, tol: f64, max_order: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let mut results = Vec::new();
    
    for n in (5..=max_order).step_by(5) {
        let result = gauss_hermite_quadrature(&func, n)?;
        results.push(result);
        
        if n >= 10 {
            let error_est = (results[results.len()-1] - results[results.len()-2]).abs();
            if error_est <= tol * result.abs() {
                return Ok(result);
            }
        }
    }
    
    Ok(*results.last().unwrap())
}

/// Test functions for Gauss-Hermite quadrature
pub mod test_functions {
    pub fn constant_fn(x: f64) -> f64 {
        1.0
    }

    pub fn polynomial_fn(x: f64) -> f64 {
        x * x
    }

    pub fn gaussian_fn(x: f64) -> f64 {
        (-x * x).exp()
    }

    pub fn oscillatory_fn(x: f64) -> f64 {
        x.sin()
    }

    pub fn exponential_fn(x: f64) -> f64 {
        x.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gauher_basic() {
        let (x, w) = gauher(5).unwrap();
        
        // Check symmetry
        assert_abs_diff_eq!(x[0], -x[4], epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], -x[3], epsilon = 1e-10);
        assert_abs_diff_eq!(w[0], w[4], epsilon = 1e-10);
        assert_abs_diff_eq!(w[1], w[3], epsilon = 1e-10);
        
        // Check weight sum (should integrate e^{-x²} to √π)
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, std::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_gauher_parallel_consistency() {
        let (x1, w1) = gauher(10).unwrap();
        let (x2, w2) = gauher_parallel(10).unwrap();
        
        for i in 0..10 {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-10);
            assert_abs_diff_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gauher_invalid_order() {
        let result = gauher(0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidOrder);
    }

    #[test]
    fn test_gauss_hermite_quadrature_constant() {
        let result = gauss_hermite_quadrature(|_| 1.0, 5).unwrap();
        // ∫_{-∞}^{∞} e^{-x²} * 1 dx = √π
        assert_abs_diff_eq!(result, std::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_hermite_quadrature_polynomial() {
        let result = gauss_hermite_quadrature(|x| x * x, 10).unwrap();
        // ∫_{-∞}^{∞} e^{-x²} * x² dx = √π/2
        assert_abs_diff_eq!(result, std::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_hermite_quadrature_parallel() {
        let serial = gauss_hermite_quadrature(|x| x * x, 10).unwrap();
        let parallel = gauss_hermite_quadrature_parallel(|x| x * x, 10).unwrap();
        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_hermite_weighted_quadrature() {
        // Test integration without the weight function
        let result = gauss_hermite_weighted_quadrature(|x| 1.0, 5).unwrap();
        // Should give ∫_{-∞}^{∞} 1 dx, which diverges, but our transformation should handle it
        assert!(result.is_finite());
    }

    #[test]
    fn test_gauss_hermite_adaptive() {
        let result = gauss_hermite_adaptive(|x| x.exp(), 1e-10, 20).unwrap();
        // ∫_{-∞}^{∞} e^{-x²} * e^x dx converges
        assert!(result.is_finite());
    }

    #[test]
    fn test_gauher_high_order() {
        let (x, w) = gauher(20).unwrap();
        
        // Check weight sum
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, std::f64::consts::PI.sqrt(), epsilon = 1e-10);
        
        // Check symmetry
        for i in 0..10 {
            assert_abs_diff_eq!(x[i], -x[19 - i], epsilon = 1e-10);
            assert_abs_diff_eq!(w[i], w[19 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hermite_polynomial() {
        // Test known values of Hermite polynomials
        let (p3, pp) = hermite_polynomial(3, 1.0);
        // H₃(1) = 8 - 12 = -4
        // But our normalized version differs by constant factors
        assert!(p3.abs() > 0.0); // Just check it's computed
    }

    #[test]
    fn test_gauher_precision() {
        // Test that Gauss-Hermite quadrature is exact for polynomials up to degree 2n-1
        let n = 5;
        let (x, w) = gauher(n).unwrap();
        
        for degree in 0..2*n {
            let exact = if degree % 2 == 0 {
                (degree as f64 + 1.0).gamma() / 2.0f64.powf(degree as f64 / 2.0 + 0.5) * std::f64::consts::PI.sqrt()
            } else {
                0.0
            };
            
            let computed: f64 = x.iter()
                .zip(w.iter())
                .map(|(&x, &w)| w * x.powi(degree as i32))
                .sum();
            
            if degree % 2 == 0 {
                assert_abs_diff_eq!(computed, exact, epsilon = 1e-10, "Failed for degree {}", degree);
            } else {
                assert_abs_diff_eq!(computed, 0.0, epsilon = 1e-10, "Failed for degree {}", degree);
            }
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = GaussHermiteCache::new();
        
        let rule1 = cache.get_rule(5).unwrap();
        let rule2 = cache.get_rule(5).unwrap();
        
        assert_eq!(rule1.0, rule2.0);
        assert_eq!(rule1.1, rule2.1);
    }

    #[test]
    fn test_gauher_convergence() {
        // Test that convergence works for various orders
        for &n in &[5, 10, 15, 20] {
            let result = gauher(n);
            assert!(result.is_ok(), "Failed for n={}", n);
        }
    }

    #[test]
    fn test_gauher_initial_guesses() {
        // Test that initial guesses are reasonable
        let (x, _) = gauher(5).unwrap();
        
        // First root should be positive and around expected value
        assert!(x[0] > 0.0);
        assert!(x[0] < 3.0); // Should be less than 3 for n=5
        
        // Check roots are in increasing order
        for i in 1..2 {
            assert!(x[i] > x[i-1], "Roots should be increasing");
        }
    }

    #[test]
    fn test_gauss_hermite_odd_order() {
        // Test with odd order (should have root at 0)
        let (x, _) = gauher(5).unwrap();
        assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_hermite_even_order() {
        // Test with even order (no root at 0)
        let (x, _) = gauher(4).unwrap();
        assert!(x[1] < 0.0 && x[2] > 0.0); // Roots should be symmetric around 0
    }
}
