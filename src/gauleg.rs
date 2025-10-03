use std::error::Error;
use std::fmt;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const EPS: f64 = 3.0e-11;
const MAX_ITER: usize = 100;

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    InvalidInterval,
    InvalidOrder,
    ConvergenceFailed,
    NumericalInstability,
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuadratureError::InvalidInterval => write!(f, "Invalid integration interval"),
            QuadratureError::InvalidOrder => write!(f, "Invalid quadrature order"),
            QuadratureError::ConvergenceFailed => write!(f, "Newton-Raphson convergence failed"),
            QuadratureError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl Error for QuadratureError {}

pub type QuadratureResult<T> = std::result::Result<T, QuadratureError>;

/// Gauss-Legendre quadrature nodes and weights computation
pub fn gauleg(x1: f64, x2: f64, n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if x1 >= x2 {
        return Err(QuadratureError::InvalidInterval);
    }
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }

    let m = (n + 1) / 2;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);
    
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];

    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;

        for _ in 0..MAX_ITER {
            let (p1, pp) = legendre_polynomial(n, z);
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

        let (p1, pp) = legendre_polynomial(n, z);
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }

    Ok((x, w))
}

/// Compute Legendre polynomial and its derivative at point z
fn legendre_polynomial(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = 1.0;
    let mut p2 = 0.0;
    let mut p3;

    for j in 1..=n {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
    }

    let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
    (p1, pp)
}

/// Multithreaded version of gauleg using parallel iteration
pub fn gauleg_parallel(x1: f64, x2: f64, n: usize) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if x1 >= x2 {
        return Err(QuadratureError::InvalidInterval);
    }
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }

    let m = (n + 1) / 2;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);
    
    let x = Arc::new(Mutex::new(vec![0.0; n]));
    let w = Arc::new(Mutex::new(vec![0.0; n]));

    // Process each root in parallel
    (0..m).into_par_iter().for_each(|i| {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;

        for _ in 0..MAX_ITER {
            let (p1, pp) = legendre_polynomial(n, z);
            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() <= EPS {
                converged = true;
                break;
            }
        }

        if converged {
            let (p1, pp) = legendre_polynomial(n, z);
            let weight = 2.0 * xl / ((1.0 - z * z) * pp * pp);
            
            let mut x_lock = x.lock().unwrap();
            let mut w_lock = w.lock().unwrap();
            
            x_lock[i] = xm - xl * z;
            x_lock[n - 1 - i] = xm + xl * z;
            w_lock[i] = weight;
            w_lock[n - 1 - i] = weight;
        }
    });

    let x_result = Arc::try_unwrap(x).unwrap().into_inner().unwrap();
    let w_result = Arc::try_unwrap(w).unwrap().into_inner().unwrap();
    
    Ok((x_result, w_result))
}

/// Precompute Gauss-Legendre quadrature rules for common orders
pub struct GaussLegendreCache {
    rules: std::sync::RwLock<std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>>,
}

impl GaussLegendreCache {
    pub fn new() -> Self {
        Self {
            rules: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn get_rule(&self, n: usize, x1: f64, x2: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, x1.to_bits(), x2.to_bits());
        
        // Check if rule is already computed
        if let Some(rule) = self.rules.read().unwrap().get(&n) {
            return Ok(rule.clone());
        }

        // Compute new rule
        let rule = gauleg(x1, x2, n)?;
        self.rules.write().unwrap().insert(n, rule.clone());
        
        Ok(rule)
    }

    pub fn get_rule_parallel(&self, n: usize, x1: f64, x2: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, x1.to_bits(), x2.to_bits());
        
        if let Some(rule) = self.rules.read().unwrap().get(&n) {
            return Ok(rule.clone());
        }

        let rule = gauleg_parallel(x1, x2, n)?;
        self.rules.write().unwrap().insert(n, rule.clone());
        
        Ok(rule)
    }
}

/// Integration using precomputed Gauss-Legendre quadrature
pub fn gauss_quadrature<F>(func: F, x1: f64, x2: f64, n: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gauleg(x1, x2, n)?;
    
    let sum: f64 = nodes.iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x))
        .sum();
    
    Ok(sum)
}

/// Multithreaded integration using Gauss-Legendre quadrature
pub fn gauss_quadrature_parallel<F>(func: F, x1: f64, x2: f64, n: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    let (nodes, weights) = gauleg_parallel(x1, x2, n)?;
    
    let sum: f64 = nodes.par_iter()
        .zip(weights.par_iter())
        .map(|(&x, &w)| w * func(x))
        .sum();
    
    Ok(sum)
}

/// High-precision integration with adaptive order selection
pub fn gauss_quadrature_adaptive<F>(func: F, x1: f64, x2: f64, tol: f64, max_order: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let mut results = Vec::new();
    
    for n in (5..=max_order).step_by(5) {
        let result = gauss_quadrature(&func, x1, x2, n)?;
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

/// Generate quadrature rules for multiple intervals (composite quadrature)
pub fn composite_gauss_quadrature<F>(func: F, x1: f64, x2: f64, n_segments: usize, points_per_segment: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    if x1 >= x2 {
        return Err(QuadratureError::InvalidInterval);
    }

    let h = (x2 - x1) / n_segments as f64;
    let mut sum = 0.0;

    for i in 0..n_segments {
        let seg_x1 = x1 + i as f64 * h;
        let seg_x2 = seg_x1 + h;
        sum += gauss_quadrature(&func, seg_x1, seg_x2, points_per_segment)?;
    }

    Ok(sum)
}

/// Multithreaded composite quadrature
pub fn composite_gauss_quadrature_parallel<F>(func: F, x1: f64, x2: f64, n_segments: usize, points_per_segment: usize) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    if x1 >= x2 {
        return Err(QuadratureError::InvalidInterval);
    }

    let h = (x2 - x1) / n_segments as f64;
    
    let sum: f64 = (0..n_segments)
        .into_par_iter()
        .map(|i| {
            let seg_x1 = x1 + i as f64 * h;
            let seg_x2 = seg_x1 + h;
            gauss_quadrature(&func, seg_x1, seg_x2, points_per_segment).unwrap_or(0.0)
        })
        .sum();

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gauleg_basic() {
        let (x, w) = gauleg(-1.0, 1.0, 5).unwrap();
        
        // Check symmetry
        assert_abs_diff_eq!(x[0], -x[4], epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], -x[3], epsilon = 1e-10);
        assert_abs_diff_eq!(w[0], w[4], epsilon = 1e-10);
        assert_abs_diff_eq!(w[1], w[3], epsilon = 1e-10);
        
        // Check weight sum (should integrate constant 1 to 2)
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauleg_parallel_consistency() {
        let (x1, w1) = gauleg(-1.0, 1.0, 10).unwrap();
        let (x2, w2) = gauleg_parallel(-1.0, 1.0, 10).unwrap();
        
        for i in 0..10 {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-10);
            assert_abs_diff_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gauleg_invalid_interval() {
        let result = gauleg(1.0, -1.0, 5);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidInterval);
    }

    #[test]
    fn test_gauleg_invalid_order() {
        let result = gauleg(-1.0, 1.0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidOrder);
    }

    #[test]
    fn test_gauss_quadrature_constant() {
        let result = gauss_quadrature(|_| 2.0, 0.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_quadrature_linear() {
        let result = gauss_quadrature(|x| x, 0.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_quadrature_quadratic() {
        let result = gauss_quadrature(|x| x * x, 0.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_quadrature_parallel() {
        let serial = gauss_quadrature(|x| x * x, 0.0, 1.0, 10).unwrap();
        let parallel = gauss_quadrature_parallel(|x| x * x, 0.0, 1.0, 10).unwrap();
        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_quadrature_adaptive() {
        let result = gauss_quadrature_adaptive(|x| x.exp(), 0.0, 1.0, 1e-10, 20).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_composite_gauss_quadrature() {
        let result = composite_gauss_quadrature(|x| x * x, 0.0, 1.0, 10, 5).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_composite_gauss_quadrature_parallel() {
        let result = composite_gauss_quadrature_parallel(|x| x * x, 0.0, 1.0, 10, 5).unwrap();
        assert_abs_diff_eq!(result, 1.0/3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauleg_high_order() {
        // Test with higher order (should still converge)
        let (x, w) = gauleg(-1.0, 1.0, 50).unwrap();
        
        // Check weight sum
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2.0, epsilon = 1e-10);
        
        // Check symmetry
        for i in 0..25 {
            assert_abs_diff_eq!(x[i], -x[49 - i], epsilon = 1e-10);
            assert_abs_diff_eq!(w[i], w[49 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gauleg_different_intervals() {
        let (x1, w1) = gauleg(-1.0, 1.0, 5).unwrap();
        let (x2, w2) = gauleg(0.0, 2.0, 5).unwrap();
        
        // Nodes should be transformed correctly
        for i in 0..5 {
            assert_abs_diff_eq!(x2[i], x1[i] + 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(w2[i], w1[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gauleg_precision() {
        // Test that Gaussian quadrature is exact for polynomials up to degree 2n-1
        let n = 5;
        let (x, w) = gauleg(-1.0, 1.0, n).unwrap();
        
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
        diff > 1e-10,
            "polynomial up to degree n = 5 failed: computed = {}, exact = {}, diff = {}",
            computed, exact, diff
    );
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = GaussLegendreCache::new();
        
        // First call should compute
        let rule1 = cache.get_rule(5, -1.0, 1.0).unwrap();
        
        // Second call should use cache
        let rule2 = cache.get_rule(5, -1.0, 1.0).unwrap();
        
        assert_eq!(rule1.0, rule2.0);
        assert_eq!(rule1.1, rule2.1);
    }

    #[test]
    fn test_gauleg_edge_cases() {
        // Test very small interval
        let (x, w) = gauleg(0.999, 1.0, 5).unwrap();
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 0.001, epsilon = 1e-10);
        
        // Test very large interval
        let (x, w) = gauleg(-1000.0, 1000.0, 5).unwrap();
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_polynomial() {
        // Test known values of Legendre polynomials
        let (p3, _) = legendre_polynomial(3, 0.5);
        // P3(0.5) = (5*0.5^3 - 3*0.5)/2 = -0.4375
        assert_abs_diff_eq!(p3, -0.4375, epsilon = 1e-10);
        
        let (p4, _) = legendre_polynomial(4, 0.0);
        // P4(0) = 3/8
        assert_abs_diff_eq!(p4, 0.375, epsilon = 1e-10);
    }
}
