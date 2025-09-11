use std::error::Error;
use std::fmt;
use std::f64::consts::{PI, EULER};
use rayon::prelude::*;
use num_complex::Complex64;

// Constants
const EPS: f64 = 6.0e-8;
const MAXIT: usize = 100;
const FPMIN: f64 = 1.0e-30;
const TMIN: f64 = 2.0;
const PIBY2: f64 = PI / 2.0;

// Custom error type
#[derive(Debug, Clone)]
pub enum CiSiError {
    ComputationError(String),
    ConvergenceError(String),
}

impl fmt::Display for CiSiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CiSiError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            CiSiError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
        }
    }
}

impl Error for CiSiError {}

// Result type alias
pub type CiSiResult<T> = Result<T, CiSiError>;

/// Cosine and sine integrals Ci(x) and Si(x)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CiSiIntegrals {
    pub ci: f64,
    pub si: f64,
}

impl CiSiIntegrals {
    pub fn new(ci: f64, si: f64) -> Self {
        Self { ci, si }
    }
}

/// Computes cosine integral Ci(x) and sine integral Si(x) for a single value
pub fn cisi(x: f64) -> CiSiResult<CiSiIntegrals> {
    let t = x.abs();
    
    // Handle t = 0 case
    if t == 0.0 {
        return Ok(CiSiIntegrals::new(-1.0 / FPMIN, 0.0));
    }
    
    let (ci, si) = if t > TMIN {
        // Continued fraction for large t
        cisi_continued_fraction(t)?
    } else {
        // Series expansion for small t
        cisi_series_expansion(t)?
    };
    
    // Adjust sign for negative x (Si is odd, Ci is even)
    let si_adjusted = if x < 0.0 { -si } else { si };
    
    Ok(CiSiIntegrals::new(ci, si_adjusted))
}

/// Continued fraction method for Ci and Si (large t)
fn cisi_continued_fraction(t: f64) -> CiSiResult<(f64, f64)> {
    let mut b = Complex64::new(1.0, t);
    let mut c = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = Complex64::new(1.0, 0.0) / b;
    let mut h = d;

    for i in 2..=MAXIT {
        let a = -((i - 1) * (i - 1)) as f64;
        b += Complex64::new(2.0, 0.0);
        
        d = Complex64::new(1.0, 0.0) / (a * d + b);
        c = b + Complex64::new(a, 0.0) / c;
        
        let del = c * d;
        h *= del;
        
        if (del.re - 1.0).abs() + del.im.abs() < EPS {
            let phase = Complex64::new(t.cos(), -t.sin());
            h *= phase;
            
            let ci = -h.re;
            let si = PIBY2 + h.im;
            
            return Ok((ci, si));
        }
    }
    
    Err(CiSiError::ConvergenceError(
        "Continued fraction did not converge in cisi_continued_fraction".to_string()
    ))
}

/// Series expansion for Ci and Si (small t)
fn cisi_series_expansion(t: f64) -> CiSiResult<(f64, f64)> {
    if t < FPMIN.sqrt() {
        return Ok((t.ln() + EULER, t));
    }

    let mut sum = 0.0;
    let mut sums = 0.0;
    let mut sumc = 0.0;
    let mut sign = 1.0;
    let mut fact = 1.0;
    let mut odd = true;

    for k in 1..=MAXIT {
        fact *= t / k as f64;
        let term = fact / k as f64;
        
        sum += sign * term;
        let err = term / sum.abs();
        
        if odd {
            sign = -sign;
            sums = sum;
            sum = sumc;
        } else {
            sumc = sum;
            sum = sums;
        }
        
        if err < EPS {
            let ci = sumc + t.ln() + EULER;
            return Ok((ci, sums));
        }
        
        odd = !odd;
    }
    
    Err(CiSiError::ConvergenceError(
        "Series expansion did not converge in cisi_series_expansion".to_string()
    ))
}

/// Computes Ci and Si integrals for multiple values in parallel
pub fn cisi_parallel(x_values: &[f64]) -> Vec<CiSiResult<CiSiIntegrals>> {
    x_values
        .par_iter()
        .map(|&x| cisi(x))
        .collect()
}

/// Precomputed cache for Ci and Si integrals
#[derive(Clone)]
pub struct CiSiComputer {
    cache: Option<Vec<CiSiIntegrals>>,
    x_min: f64,
    x_max: f64,
    step: f64,
    x_values: Vec<f64>,
}

impl CiSiComputer {
    /// Creates a new CiSi computer
    pub fn new(x_min: f64, x_max: f64, n_points: usize) -> Self {
        let step = (x_max - x_min) / (n_points - 1) as f64;
        let x_values: Vec<f64> = (0..n_points)
            .map(|i| x_min + i as f64 * step)
            .collect();
        
        Self {
            cache: None,
            x_min,
            x_max,
            step,
            x_values,
        }
    }
    
    /// Precomputes values for the entire range
    pub fn precompute(&mut self) -> CiSiResult<()> {
        let results = cisi_parallel(&self.x_values);
        
        // Check for any errors
        if let Some(err) = results.iter().find_map(|r| r.as_ref().err()) {
            return Err(CiSiError::ComputationError(
                format!("Precomputation failed: {}", err)
            ));
        }
        
        self.cache = Some(results.into_iter().map(|r| r.unwrap()).collect());
        Ok(())
    }
    
    /// Gets Ci and Si integrals for a specific x value
    pub fn get(&self, x: f64) -> CiSiResult<CiSiIntegrals> {
        // Use cache for values within the precomputed range
        if let Some(cache) = &self.cache {
            if x >= self.x_min && x <= self.x_max {
                let idx = ((x - self.x_min) / self.step).round() as usize;
                if idx < cache.len() {
                    return Ok(cache[idx]);
                }
            }
        }
        
        // Fall back to direct computation
        cisi(x)
    }
    
    /// Gets all precomputed x values
    pub fn x_values(&self) -> &[f64] {
        &self.x_values
    }
    
    /// Gets all precomputed results (if available)
    pub fn results(&self) -> Option<&[CiSiIntegrals]> {
        self.cache.as_deref()
    }
}

/// Additional utility functions

/// Computes the exponential integral E₁(x) = -Ei(-x) for x > 0
pub fn exponential_integral_e1(x: f64) -> CiSiResult<f64> {
    if x <= 0.0 {
        return Err(CiSiError::ComputationError(
            "Exponential integral E₁ is defined only for x > 0".to_string()
        ));
    }
    
    // For small x, use series expansion
    if x <= 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0;
        let mut sign = -1.0;
        let mut fact = 1.0;
        
        for k in 1..=MAXIT {
            term *= x / k as f64;
            let old_sum = sum;
            sum += sign * term / k as f64;
            
            if (sum - old_sum).abs() < EPS * sum.abs() {
                return Ok(-EULER - x.ln() + sum);
            }
            
            sign = -sign;
            fact += 1.0;
        }
        
        return Err(CiSiError::ConvergenceError(
            "Series expansion did not converge for E₁".to_string()
        ));
    }
    
    // For large x, use continued fraction
    let mut b = x + 1.0;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..=MAXIT {
        let a = - (i as f64) * (i as f64);
        b += 2.0;
        
        d = 1.0 / (a * d + b);
        c = b + a / c;
        
        let del = c * d;
        h *= del;
        
        if del.abs().abs_diff(1.0) < EPS {
            return Ok((-x).exp() * h);
        }
    }
    
    Err(CiSiError::ConvergenceError(
        "Continued fraction did not converge for E₁".to_string()
    ))
}

/// Computes the entire exponential integral function
pub fn exponential_integral(x: f64) -> CiSiResult<f64> {
    if x > 0.0 {
        // Ei(x) = -E₁(-x) for x > 0
        exponential_integral_e1(-x).map(|e1| -e1)
    } else if x < 0.0 {
        // For negative x, use the principal value
        exponential_integral_e1(-x)
    } else {
        Err(CiSiError::ComputationError(
            "Exponential integral is undefined at x = 0".to_string()
        ))
    }
}

/// Benchmarking structure for performance monitoring
pub struct CiSiBenchmark {
    pub times: Vec<std::time::Duration>,
    pub errors: Vec<f64>,
}

impl CiSiBenchmark {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            errors: Vec::new(),
        }
    }
    
    /// Benchmarks the CiSi computation
    pub fn benchmark(&mut self, x_values: &[f64], samples: usize) {
        self.times.clear();
        self.errors.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let results: Vec<_> = x_values
                .par_iter()
                .map(|&x| cisi(x))
                .collect();
            let duration = start.elapsed();
            
            self.times.push(duration);
            
            // Calculate error rate
            let error_rate = results.iter()
                .filter_map(|r| r.as_ref().err().map(|_| 1.0))
                .sum::<f64>() / x_values.len() as f64;
            
            self.errors.push(error_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_small_values() {
        // Test very small x
        let result = cisi(1e-10).unwrap();
        assert_abs_diff_eq!(result.si, 1e-10, epsilon = 1e-15);
        assert!(result.ci < 0.0); // Ci should be large negative
        
        // Test small positive x
        let result = cisi(0.5).unwrap();
        assert_abs_diff_eq!(result.si, 0.493107, epsilon = 1e-6);
        assert_abs_diff_eq!(result.ci, -0.177784, epsilon = 1e-6);
    }

    #[test]
    fn test_zero() {
        let result = cisi(0.0).unwrap();
        assert_abs_diff_eq!(result.si, 0.0, epsilon = 1e-15);
        assert!(result.ci < -1e30); // Should be very large negative
    }

    #[test]
    fn test_negative_values() {
        let pos = cisi(1.0).unwrap();
        let neg = cisi(-1.0).unwrap();
        
        // Si is odd function
        assert_abs_diff_eq!(pos.si, -neg.si, epsilon = 1e-15);
        // Ci is even function
        assert_abs_diff_eq!(pos.ci, neg.ci, epsilon = 1e-15);
    }

    #[test]
    fn test_large_values() {
        // Test large positive x
        let result = cisi(10.0).unwrap();
        assert_abs_diff_eq!(result.si, 1.658347, epsilon = 1e-6);
        assert_abs_diff_eq!(result.ci, -0.045456, epsilon = 1e-6);
        
        // Test very large x
        let result = cisi(100.0).unwrap();
        assert_abs_diff_eq!(result.si, 1.562225, epsilon = 1e-6);
        assert_abs_diff_eq!(result.ci, -0.005149, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_computation() {
        let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let results = cisi_parallel(&x_values);
        
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));
        
        // Verify function properties
        let pos_1 = results[3].as_ref().unwrap();
        let neg_1 = results[1].as_ref().unwrap();
        assert_abs_diff_eq!(pos_1.si, -neg_1.si, epsilon = 1e-15);
        assert_abs_diff_eq!(pos_1.ci, neg_1.ci, epsilon = 1e-15);
    }

    #[test]
    fn test_exponential_integral() {
        let e1 = exponential_integral_e1(1.0).unwrap();
        assert_abs_diff_eq!(e1, 0.219384, epsilon = 1e-6);
        
        let ei = exponential_integral(1.0).unwrap();
        assert_abs_diff_eq!(ei, 1.895117, epsilon = 1e-6);
    }
}
