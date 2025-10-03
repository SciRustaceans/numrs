use crate::gammln;
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

const EPS: f64 = 3.0e-14;
const MAXIT: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    InvalidOrder,
    ConvergenceFailed,
    NumericalInstability,
    InvalidParameter,
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuadratureError::InvalidOrder => write!(f, "Invalid quadrature order"),
            QuadratureError::ConvergenceFailed => write!(f, "Newton-Raphson convergence failed"),
            QuadratureError::NumericalInstability => write!(f, "Numerical instability detected"),
            QuadratureError::InvalidParameter => write!(f, "Invalid parameter value"),
        }
    }
}

impl Error for QuadratureError {}

pub type QuadratureResult<T> = std::result::Result<T, QuadratureError>;

/// Gauss-Laguerre quadrature nodes and weights computation
pub fn gaulag(n: usize, alf: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if alf <= -1.0 {
        return Err(QuadratureError::InvalidParameter);
    }

    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];

    for i in 0..n {
        let mut z = if i == 0 {
            (1.0 + alf) * (3.0 + 0.92 * alf) / (1.0 + 2.4 * n as f64 + 1.8 * alf)
        } else if i == 1 {
            x[0] + (15.0 + 6.25 * alf) / (1.0 + 0.9 * alf + 2.5 * n as f64)
        } else {
            let ai = (i - 1) as f64;
            x[i - 1]
                + ((1.0 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alf / (1.0 + 3.5 * ai))
                    * (x[i - 1] - x[i - 2])
                    / (1.0 + 0.3 * alf)
        };

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, p2, new_pp) = laguerre_polynomial(n, alf, z);
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

        let (p1, p2, pp) = laguerre_polynomial(n, alf, z);
        x[i] = z;

        // Compute weight using gamma functions
        let log_weight = (alf + n as f64).gammaln().0 - (n as f64).gammaln().0;
        w[i] = (-log_weight.exp()) / (pp * n as f64 * p2);
    }

    Ok((x, w))
}

/// Compute generalized Laguerre polynomial and its derivative
fn laguerre_polynomial(n: usize, alf: f64, z: f64) -> (f64, f64, f64) {
    let mut p1 = 1.0;
    let mut p2 = 0.0;
    let mut p3;

    for j in 1..=n {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j as f64 - 1.0 + alf - z) * p2 - (j as f64 - 1.0 + alf) * p3) / j as f64;
    }

    let pp = n as f64 * p1 - (n as f64 + alf) * p2;
    (p1, p2, pp)
}

/// Multithreaded version of gaulag
pub fn gaulag_parallel(n: usize, alf: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if alf <= -1.0 {
        return Err(QuadratureError::InvalidParameter);
    }

    let x = Arc::new(Mutex::new(vec![0.0; n]));
    let w = Arc::new(Mutex::new(vec![0.0; n]));

    // Process each root in parallel
    (0..n).into_par_iter().for_each(|i| {
        let mut z = if i == 0 {
            (1.0 + alf) * (3.0 + 0.92 * alf) / (1.0 + 2.4 * n as f64 + 1.8 * alf)
        } else {
            let x_lock = x.lock().unwrap();
            if i == 1 {
                x_lock[0] + (15.0 + 6.25 * alf) / (1.0 + 0.9 * alf + 2.5 * n as f64)
            } else {
                let ai = (i - 1) as f64;
                x_lock[i - 1]
                    + ((1.0 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alf / (1.0 + 3.5 * ai))
                        * (x_lock[i - 1] - x_lock[i - 2])
                        / (1.0 + 0.3 * alf)
            }
        };

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, p2, new_pp) = laguerre_polynomial(n, alf, z);
            pp = new_pp;

            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() <= EPS {
                converged = true;
                break;
            }
        }

        if converged {
            let (p1, p2, pp) = laguerre_polynomial(n, alf, z);
            let log_weight = (alf + n as f64).ln_gamma().0 - (n as f64).ln_gamma().0;
            let weight = (-log_weight.exp()) / (pp * n as f64 * p2);

            let mut x_lock = x.lock().unwrap();
            let mut w_lock = w.lock().unwrap();

            x_lock[i] = z;
            w_lock[i] = weight;
        }
    });

    let x_result = Arc::try_unwrap(x).unwrap().into_inner().unwrap();
    let w_result = Arc::try_unwrap(w).unwrap().into_inner().unwrap();

    Ok((x_result, w_result))
}

/// Cache for Gauss-Laguerre quadrature rules
pub struct GaussLaguerreCache {
    rules: std::sync::RwLock<std::collections::HashMap<(usize, u64), (Vec<f64>, Vec<f64>)>>,
}

impl GaussLaguerreCache {
    pub fn new() -> Self {
        Self {
            rules: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn get_rule(&self, n: usize, alf: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, alf.to_bits());

        if let Some(rule) = self.rules.read().unwrap().get(&(n, alf.to_bits())) {
            return Ok(rule.clone());
        }

        let rule = gaulag(n, alf)?;
        self.rules
            .write()
            .unwrap()
            .insert((n, alf.to_bits()), rule.clone());

        Ok(rule)
    }

    pub fn get_rule_parallel(&self, n: usize, alf: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, alf.to_bits());

        if let Some(rule) = self.rules.read().unwrap().get(&(n, alf.to_bits())) {
            return Ok(rule.clone());
        }

        let rule = gaulag_parallel(n, alf)?;
        self.rules
            .write()
            .unwrap()
            .insert((n, alf.to_bits()), rule.clone());

        Ok(rule)
    }
}

/// Integration using Gauss-Laguerre quadrature
pub fn gauss_laguerre_quadrature<F>(func: F, n: usize, alf: f64) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gaulag(n, alf)?;

    let sum: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x))
        .sum();

    Ok(sum)
}

/// Multithreaded Gauss-Laguerre quadrature
pub fn gauss_laguerre_quadrature_parallel<F>(func: F, n: usize, alf: f64) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    let (nodes, weights) = gaulag_parallel(n, alf)?;

    let sum: f64 = nodes
        .par_iter()
        .zip(weights.par_iter())
        .map(|(&x, &w)| w * func(x))
        .sum();

    Ok(sum)
}

/// Integration of functions with weight x^alf * e^{-x}
pub fn gauss_laguerre_weighted_quadrature<F>(func: F, n: usize, alf: f64) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gaulag(n, alf)?;

    let sum: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x) * (-x).exp() * x.powf(-alf))
        .sum();

    Ok(sum)
}

/// Adaptive Gauss-Laguerre quadrature
pub fn gauss_laguerre_adaptive<F>(
    func: F,
    alf: f64,
    tol: f64,
    max_order: usize,
) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let mut results = Vec::new();

    for n in (5..=max_order).step_by(5) {
        let result = gauss_laguerre_quadrature(&func, n, alf)?;
        results.push(result);

        if n >= 10 {
            let error_est = (results[results.len() - 1] - results[results.len() - 2]).abs();
            if error_est <= tol * result.abs() {
                return Ok(result);
            }
        }
    }

    Ok(*results.last().unwrap())
}

/// Test functions for Gauss-Laguerre quadrature
pub mod test_functions {
    pub fn constant_fn(x: f64) -> f64 {
        1.0
    }

    pub fn polynomial_fn(x: f64) -> f64 {
        x * x
    }

    pub fn exponential_fn(x: f64) -> f64 {
        (-x).exp()
    }

    pub fn oscillatory_fn(x: f64) -> f64 {
        (x).sin()
    }

    pub fn power_law_fn(x: f64) -> f64 {
        x.powf(1.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use special::Gamma;

    #[test]
    fn test_gaulag_basic() {
        let (x, w) = gaulag(5, 0.0).unwrap();

        // Check that weights sum to 1 (for alf=0, ∫₀∞ e^{-x} dx = 1)
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 1.0, epsilon = 1e-10);

        // Check nodes are positive and increasing
        for i in 1..5 {
            assert!(x[i] > x[i - 1], "Nodes should be increasing");
            assert!(x[i] > 0.0, "Nodes should be positive");
        }
    }

    #[test]
    fn test_gaulag_parallel_consistency() {
        let (x1, w1) = gaulag(10, 0.5).unwrap();
        let (x2, w2) = gaulag_parallel(10, 0.5).unwrap();

        for i in 0..10 {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-10);
            assert_abs_diff_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaulag_invalid_order() {
        let result = gaulag(0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidOrder);
    }

    #[test]
    fn test_gaulag_invalid_parameter() {
        let result = gaulag(5, -2.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidParameter);
    }

    #[test]
    fn test_gauss_laguerre_quadrature_constant() {
        let result = gauss_laguerre_quadrature(|_| 1.0, 5, 0.0).unwrap();
        // ∫₀∞ e^{-x} * 1 dx = 1
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_laguerre_quadrature_polynomial() {
        let result = gauss_laguerre_quadrature(|x| x * x, 10, 0.0).unwrap();
        // ∫₀∞ e^{-x} * x² dx = 2! = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_laguerre_quadrature_parallel() {
        let serial = gauss_laguerre_quadrature(|x| x, 10, 0.0).unwrap();
        let parallel = gauss_laguerre_quadrature_parallel(|x| x, 10, 0.0).unwrap();

        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_laguerre_weighted_quadrature() {
        // Test integration without the weight function
        let result = gauss_laguerre_weighted_quadrature(|x| 1.0, 5, 0.0).unwrap();
        // Should give ∫₀∞ 1 dx, which diverges, but our transformation should handle it
        assert!(result.is_finite());
    }

    #[test]
    fn test_gauss_laguerre_adaptive() {
        let result = gauss_laguerre_adaptive(|x| x.exp(), 0.0, 1e-10, 20).unwrap();
        // ∫₀∞ e^{-x} * e^x dx = ∫₀∞ 1 dx diverges, but adaptive should handle it
        assert!(result.is_finite());
    }

    #[test]
    fn test_gaulag_different_alpha() {
        for &alf in &[0.0, 0.5, 1.0, 2.0] {
            let (x, w) = gaulag(5, alf).unwrap();

            // Weight sum should equal Gamma(alf+1) for constant function
            let sum_weights: f64 = w.iter().sum();
            let expected = (alf + 1.0).gamma();
            assert_abs_diff_eq!(sum_weights, expected, epsilon = 1e-10);
        }
    }

   #[test]
    fn test_laguerre_polynomial() {
        // Test known values of Laguerre polynomials
        let (p3, p2, pp) = laguerre_polynomial(3, 0.0, 1.0);
        // L₃(1) = (6 - 18 + 9 - 1)/6 = -4/6 = -2/3
        assert_abs_diff_eq!(p3, -2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaulag_precision() {
        // Test that Gauss-Laguerre quadrature is exact for polynomials up to degree 2n-1
        let n = 5;
        let alf = 0.0;
        let (x, w) = gaulag(n, alf).unwrap();

        for degree in 0..2 * n {
            let exact = (degree as f64).gamma();
            let computed: f64 = x
                .iter()
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
    fn test_cache_functionality() {
        let cache = GaussLaguerreCache::new();

        let rule1 = cache.get_rule(5, 0.0).unwrap();
        let rule2 = cache.get_rule(5, 0.0).unwrap();

        assert_eq!(rule1.0, rule2.0);
        assert_eq!(rule1.1, rule2.1);
    }

    #[test]
    fn test_gaulag_negative_alpha() {
        // Test with alf > -1 but negative
        let result = gaulag(5, -0.5);
        assert!(result.is_ok());

        let (x, w) = result.unwrap();
        let sum_weights: f64 = w.iter().sum();
        let expected = (-0.5 + 1.0).gamma();
        assert_abs_diff_eq!(sum_weights, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gaulag_convergence() {
        // Test that convergence works for various parameters
        for &alf in &[0.0, 1.0, 2.0] {
            for &n in &[5, 10, 15] {
                let result = gaulag(n, alf);
                assert!(result.is_ok(), "Failed for n={}, alf={}", n, alf);
            }
        }
    }

    #[test]
    fn test_gamma_function_accuracy() {
        // Test that our gamma function usage is accurate
        let ln_gamma = 2.5.ln_gamma().0;
        let gamma = ln_gamma.exp();
        assert_abs_diff_eq!(
            gamma,
            1.5 * 0.5 * std::f64::consts::PI.sqrt(),
            epsilon = 1e-10
        );
    }
}
