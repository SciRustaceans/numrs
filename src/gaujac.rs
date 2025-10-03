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
    InvalidParameters,
    ConvergenceFailed,
    NumericalInstability,
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuadratureError::InvalidOrder => write!(f, "Invalid quadrature order"),
            QuadratureError::InvalidParameters => {
                write!(f, "Invalid parameters (alpha, beta <= -1)")
            }
            QuadratureError::ConvergenceFailed => write!(f, "Newton-Raphson convergence failed"),
            QuadratureError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl Error for QuadratureError {}

pub type QuadratureResult<T> = std::result::Result<T, QuadratureError>;

/// Gauss-Jacobi quadrature nodes and weights computation
pub fn gaujac(n: usize, alf: f64, bet: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if alf <= -1.0 || bet <= -1.0 {
        return Err(QuadratureError::InvalidParameters);
    }

    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let alfbet = alf + bet;

    for i in 0..n {
        let mut z = if i == 0 {
            let an = alf / n as f64;
            let bn = bet / n as f64;
            let r1 = (1.0 + alf) * (2.78 / (4.0 + (n * n) as f64) + 0.768 * an / n as f64);
            let r2 = 1.0 + 1.48 * an + 0.96 * bn + 0.452 * an * an + 0.83 * an * bn;
            1.0 - r1 / r2
        } else if i == 1 {
            let r1 = (4.1 + alf) / ((1.0 + alf) * (1.0 + 0.56 * alf));
            let r2 = 1.0 + 0.06 * (n as f64 - 8.0) * (1.0 + 0.12 * alf) / n as f64;
            let r3 = 1.0 + 8.0 * bet / ((6.28 + bet) * (n * n) as f64);
            x[0] - (x[0] - x[0]) * r1 * r2 * r3 // x[0] - z, but z is x[0] in this case
        } else if i == n - 1 {
            let r1 = (1.0 + 0.235 * bet) / (0.766 + 0.119 * bet);
            let r2 = 1.0 / (1.0 + 0.639 * (n as f64 - 4.0) / (1.0 + 0.71 * (n as f64 - 4.0)));
            let r3 = 1.0 / (1.0 + 20.0 * alf / ((7.5 + alf) * (n * n) as f64));
            x[n - 2] + (x[n - 2] - x[n - 3]) * r1 * r2 * r3
        } else if i == n - 2 {
            let r1 = (1.0 + 0.37 * bet) / (1.67 + 0.28 * bet);
            let r2 = 1.0 / (1.0 + 0.22 * (n as f64 - 8.0) / n as f64);
            let r3 = 1.0 / (1.0 + 8.0 * alf / ((6.28 + alf) * (n * n) as f64));
            x[n - 3] + (x[n - 3] - x[n - 4]) * r1 * r2 * r3
        } else {
            3.0 * x[i - 1] - 3.0 * x[i - 2] + x[i - 3]
        };

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, p2, new_pp) = jacobi_polynomial(n, alf, bet, z);
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

        let (p1, p2, pp) = jacobi_polynomial(n, alf, bet, z);
        x[i] = z;

        // Compute weight using gamma functions
        let log_num = (alf + n as f64).gammaln().0 + (bet + n as f64).gammaln().0;
        let log_den = (n as f64 + 1.0).gammaln().0 + (n as f64 + alfbet + 1.0).gammaln().0;
        let temp = 2.0 * n as f64 + alfbet;
        let weight = (log_num - log_den).exp() * temp * 2.0f64.powf(alfbet) / (pp * p2);

        w[i] = weight;
    }

    Ok((x, w))
}

/// Compute Jacobi polynomial and its derivative
fn jacobi_polynomial(n: usize, alf: f64, bet: f64, z: f64) -> (f64, f64, f64) {
    let alfbet = alf + bet;
    let mut p1 = (alf - bet + (2.0 + alfbet) * z) / 2.0;
    let mut p2 = 1.0;
    let mut p3;

    for j in 2..=n {
        p3 = p2;
        p2 = p1;
        let temp = 2.0 * j as f64 + alfbet;
        let a = 2.0 * j as f64 * (j as f64 + alfbet) * (temp - 2.0);
        let b = (temp - 1.0) * (alf * alf - bet * bet + temp * (temp - 2.0) * z);
        let c = 2.0 * (j as f64 - 1.0 + alf) * (j as f64 - 1.0 + bet) * temp;
        p1 = (b * p2 - c * p3) / a;
    }

    let temp = 2.0 * n as f64 + alfbet;
    let pp = (n as f64 * (alf - bet - temp * z) * p1
        + 2.0 * (n as f64 + alf) * (n as f64 + bet) * p2)
        / (temp * (1.0 - z * z));

    (p1, p2, pp)
}

/// Multithreaded version of gaujac
pub fn gaujac_parallel(n: usize, alf: f64, bet: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QuadratureError::InvalidOrder);
    }
    if alf <= -1.0 || bet <= -1.0 {
        return Err(QuadratureError::InvalidParameters);
    }

    let x = Arc::new(Mutex::new(vec![0.0; n]));
    let w = Arc::new(Mutex::new(vec![0.0; n]));
    let alfbet = alf + bet;

    // Process each root in parallel
    (0..n).into_par_iter().for_each(|i| {
        let x_lock = x.lock().unwrap();
        let mut z = if i == 0 {
            let an = alf / n as f64;
            let bn = bet / n as f64;
            let r1 = (1.0 + alf) * (2.78 / (4.0 + (n * n) as f64) + 0.768 * an / n as f64);
            let r2 = 1.0 + 1.48 * an + 0.96 * bn + 0.452 * an * an + 0.83 * an * bn;
            1.0 - r1 / r2
        } else if i == 1 {
            let r1 = (4.1 + alf) / ((1.0 + alf) * (1.0 + 0.56 * alf));
            let r2 = 1.0 + 0.06 * (n as f64 - 8.0) * (1.0 + 0.12 * alf) / n as f64;
            let r3 = 1.0 + 8.0 * bet / ((6.28 + bet) * (n * n) as f64);
            x_lock[0] - (x_lock[0] - x_lock[0]) * r1 * r2 * r3
        } else if i == n - 1 {
            let r1 = (1.0 + 0.235 * bet) / (0.766 + 0.119 * bet);
            let r2 = 1.0 / (1.0 + 0.639 * (n as f64 - 4.0) / (1.0 + 0.71 * (n as f64 - 4.0)));
            let r3 = 1.0 / (1.0 + 20.0 * alf / ((7.5 + alf) * (n * n) as f64));
            x_lock[n - 2] + (x_lock[n - 2] - x_lock[n - 3]) * r1 * r2 * r3
        } else if i == n - 2 {
            let r1 = (1.0 + 0.37 * bet) / (1.67 + 0.28 * bet);
            let r2 = 1.0 / (1.0 + 0.22 * (n as f64 - 8.0) / n as f64);
            let r3 = 1.0 / (1.0 + 8.0 * alf / ((6.28 + alf) * (n * n) as f64));
            x_lock[n - 3] + (x_lock[n - 3] - x_lock[n - 4]) * r1 * r2 * r3
        } else {
            3.0 * x_lock[i - 1] - 3.0 * x_lock[i - 2] + x_lock[i - 3]
        };
        drop(x_lock);

        let mut converged = false;
        let mut pp = 0.0;

        for _ in 0..MAXIT {
            let (p1, p2, new_pp) = jacobi_polynomial(n, alf, bet, z);
            pp = new_pp;

            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() <= EPS {
                converged = true;
                break;
            }
        }

        if converged {
            let (p1, p2, pp) = jacobi_polynomial(n, alf, bet, z);
            let log_num = (alf + n as f64).ln_gamma().0 + (bet + n as f64).ln_gamma().0;
            let log_den = (n as f64 + 1.0).ln_gamma().0 + (n as f64 + alfbet + 1.0).ln_gamma().0;
            let temp = 2.0 * n as f64 + alfbet;
            let weight = (log_num - log_den).exp() * temp * 2.0f64.powf(alfbet) / (pp * p2);

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

/// Cache for Gauss-Jacobi quadrature rules
pub struct GaussJacobiCache {
    rules: std::sync::RwLock<std::collections::HashMap<(usize, u64, u64), (Vec<f64>, Vec<f64>)>>,
}

impl GaussJacobiCache {
    pub fn new() -> Self {
        Self {
            rules: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn get_rule(&self, n: usize, alf: f64, bet: f64) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, alf.to_bits(), bet.to_bits());

        if let Some(rule) = self.rules.read().unwrap().get(&key) {
            return Ok(rule.clone());
        }

        let rule = gaujac(n, alf, bet)?;
        self.rules.write().unwrap().insert(key, rule.clone());

        Ok(rule)
    }

    pub fn get_rule_parallel(
        &self,
        n: usize,
        alf: f64,
        bet: f64,
    ) -> QuadratureResult<(Vec<f64>, Vec<f64>)> {
        let key = (n, alf.to_bits(), bet.to_bits());

        if let Some(rule) = self.rules.read().unwrap().get(&key) {
            return Ok(rule.clone());
        }

        let rule = gaujac_parallel(n, alf, bet)?;
        self.rules.write().unwrap().insert(key, rule.clone());

        Ok(rule)
    }
}

/// Integration using Gauss-Jacobi quadrature
pub fn gauss_jacobi_quadrature<F>(func: F, n: usize, alf: f64, bet: f64) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gaujac(n, alf, bet)?;

    let sum: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x))
        .sum();

    Ok(sum)
}

/// Multithreaded Gauss-Jacobi quadrature
pub fn gauss_jacobi_quadrature_parallel<F>(
    func: F,
    n: usize,
    alf: f64,
    bet: f64,
) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    let (nodes, weights) = gaujac_parallel(n, alf, bet)?;

    let sum: f64 = nodes
        .par_iter()
        .zip(weights.par_iter())
        .map(|(&x, &w)| w * func(x))
        .sum();

    Ok(sum)
}

/// Integration with weight function (1-x)^alpha * (1+x)^beta
pub fn gauss_jacobi_weighted_quadrature<F>(
    func: F,
    n: usize,
    alf: f64,
    bet: f64,
) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gaujac(n, alf, bet)?;

    let sum: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * func(x) * (1.0 - x).powf(-alf) * (1.0 + x).powf(-bet))
        .sum();

    Ok(sum)
}

/// Adaptive Gauss-Jacobi quadrature
pub fn gauss_jacobi_adaptive<F>(
    func: F,
    alf: f64,
    bet: f64,
    tol: f64,
    max_order: usize,
) -> QuadratureResult<f64>
where
    F: Fn(f64) -> f64,
{
    let mut results = Vec::new();

    for n in (5..=max_order).step_by(5) {
        let result = gauss_jacobi_quadrature(&func, n, alf, bet)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gaujac_basic() {
        let (x, w) = gaujac(5, 0.0, 0.0).unwrap();

        // For alpha=beta=0, should reduce to Legendre quadrature
        // Check symmetry
        assert_abs_diff_eq!(x[0], -x[4], epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], -x[3], epsilon = 1e-10);
        assert_abs_diff_eq!(w[0], w[4], epsilon = 1e-10);
        assert_abs_diff_eq!(w[1], w[3], epsilon = 1e-10);
    }

    #[test]
    fn test_gaujac_parallel_consistency() {
        let (x1, w1) = gaujac(10, 0.5, 0.5).unwrap();
        let (x2, w2) = gaujac_parallel(10, 0.5, 0.5).unwrap();

        for i in 0..10 {
            assert_abs_diff_eq!(x1[i], x2[i], epsilon = 1e-10);
            assert_abs_diff_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaujac_invalid_order() {
        let result = gaujac(0, 0.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidOrder);
    }

    #[test]
    fn test_gaujac_invalid_parameters() {
        let result = gaujac(5, -2.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), QuadratureError::InvalidParameters);
    }

    #[test]
    fn test_gauss_jacobi_quadrature_constant() {
        let result = gauss_jacobi_quadrature(|_| 1.0, 5, 0.0, 0.0).unwrap();
        // ∫_{-1}^{1} (1-x)^0 * (1+x)^0 * 1 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_jacobi_quadrature_polynomial() {
        let result = gauss_jacobi_quadrature(|x| x * x, 10, 0.0, 0.0).unwrap();
        // ∫_{-1}^{1} x² dx = 2/3
        assert_abs_diff_eq!(result, 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_jacobi_quadrature_parallel() {
        let serial = gauss_jacobi_quadrature(|x| x * x, 10, 0.0, 0.0).unwrap();
        let parallel = gauss_jacobi_quadrature_parallel(|x| x * x, 10, 0.0, 0.0).unwrap();
        assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_jacobi_weighted_quadrature() {
        let result = gauss_jacobi_weighted_quadrature(|x| 1.0, 5, 0.0, 0.0).unwrap();
        // Should give ∫_{-1}^{1} 1 dx = 2
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaujac_different_parameters() {
        for &(alf, bet) in &[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)] {
            let (x, w) = gaujac(5, alf, bet).unwrap();

            // Weight sum should be correct
            let sum_weights: f64 = w.iter().sum();
            let expected = 2.0f64.powf(alf + bet + 1.0) * (alf + 1.0).gamma() * (bet + 1.0).gamma()
                / (alf + bet + 2.0).gamma();
            assert_abs_diff_eq!(sum_weights, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaujac_high_order() {
        let (x, w) = gaujac(20, 0.0, 0.0).unwrap();

        // Check weight sum
        let sum_weights: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum_weights, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_polynomial() {
        // Test known values of Jacobi polynomials
        let (p1, p2, pp) = jacobi_polynomial(3, 0.0, 0.0, 0.5);
        // P₃(0.5) for alpha=beta=0 (Legendre) should be (5*0.5^3 - 3*0.5)/2 = -0.4375
        assert_abs_diff_eq!(p1, -0.4375, epsilon = 1e-10);
    }

    #[test]
    fn test_gaujac_precision() {
        // Test that Gauss-Jacobi quadrature is exact for polynomials up to degree 2n-1
        let n = 5;
        let alf = 0.0;
        let bet = 0.0;
        let (x, w) = gaujac(n, alf, bet).unwrap();

        for degree in 0..2 * n {
            let exact = if degree % 2 == 0 {
                2.0 / (degree as f64 + 1.0)
            } else {
                0.0
            };

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
        let cache = GaussJacobiCache::new();

        let rule1 = cache.get_rule(5, 0.0, 0.0).unwrap();
        let rule2 = cache.get_rule(5, 0.0, 0.0).unwrap();

        assert_eq!(rule1.0, rule2.0);
        assert_eq!(rule1.1, rule2.1);
    }

    #[test]
    fn test_gaujac_convergence() {
        // Test that convergence works for various parameters
        for &(alf, bet) in &[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
            for &n in &[5, 10, 15] {
                let result = gaujac(n, alf, bet);
                assert!(
                    result.is_ok(),
                    "Failed for n={}, alf={}, bet={}",
                    n,
                    alf,
                    bet
                );
            }
        }
    }

    #[test]
    fn test_gaujac_negative_parameters() {
        // Test with parameters > -1 but negative
        let result = gaujac(5, -0.5, -0.5);
        assert!(result.is_ok());

        let (x, w) = result.unwrap();
        let sum_weights: f64 = w.iter().sum();
        let expected = 2.0f64.powf(-0.5 - 0.5 + 1.0) * (-0.5 + 1.0).gamma() * (-0.5 + 1.0).gamma()
            / (-0.5 - 0.5 + 2.0).gamma();
        assert_abs_diff_eq!(sum_weights, expected, epsilon = 1e-10);
    }
}
