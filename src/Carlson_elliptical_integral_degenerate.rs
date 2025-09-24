use std::error::Error;
use std::fmt;
use rayon::prelude::*;

// Constants
const ERRTOL: f64 = 0.04;
const TINY: f64 = 1.69e-38;
const SQRTNY: f64 = 1.3e-19;
const BIG: f64 = 3.0e37;
const TNBG: f64 = TINY * BIG;
const COMP1: f64 = 2.236 / SQRTNY;
const COMP2: f64 = (TNBG * TNBG) / 25.0;
const C1: f64 = 0.3;
const C2: f64 = 1.0 / 7.0;
const C3: f64 = 0.375;
const C4: f64 = 9.0 / 22.0;
const THIRD: f64 = 1.0 / 3.0;

// Custom error type
#[derive(Debug, Clone)]
pub enum EllipticError {
    InvalidArguments(String),
    ComputationError(String),
}

impl fmt::Display for EllipticError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EllipticError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            EllipticError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for EllipticError {}

// Result type alias
pub type EllipticResult<T> = Result<T, EllipticError>;

/// Carlson's degenerate elliptic integral RC(x, y)
/// 
/// # Arguments
/// * `x` - First parameter (must be non-negative)
/// * `y` - Second parameter (must not be zero and satisfy convergence criteria)
/// 
/// # Returns
/// The value of the integral or an error
pub fn rc(x: f64, y: f64) -> EllipticResult<f64> {
    // Input validation
    if x < 0.0 || y == 0.0 {
        return Err(EllipticError::InvalidArguments(
            "x must be non-negative and y must not be zero".to_string()
        ));
    }
    
    if x + y.abs() < TINY {
        return Err(EllipticError::InvalidArguments(
            "x + |y| is too small".to_string()
        ));
    }
    
    if x + y.abs() > BIG {
        return Err(EllipticError::InvalidArguments(
            "x + |y| is too large".to_string()
        ));
    }
    
    if y < -COMP1 && x > 0.0 && x < COMP2 {
        return Err(EllipticError::InvalidArguments(
            "Invalid combination of x and y".to_string()
        ));
    }

    let (mut xt, mut yt, w) = if y > 0.0 {
        (x, y, 1.0)
    } else {
        let xt_val = x - y;
        let yt_val = -y;
        let w_val = x.sqrt() / xt_val.sqrt();
        (xt_val, yt_val, w_val)
    };

    loop {
        let sqrt_xt = xt.sqrt();
        let sqrt_yt = yt.sqrt();
        let alamb = 2.0 * sqrt_xt * sqrt_yt + yt;
        
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        
        let ave = THIRD * (xt + yt + yt);
        let s = (yt - ave) / ave;
        
        if s.abs() <= ERRTOL {
            return Ok(w * (1.0 + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / ave.sqrt());
        }
    }
}

/// Computes RC for multiple argument pairs in parallel
pub fn rc_parallel(args: &[(f64, f64)]) -> Vec<EllipticResult<f64>> {
    args.par_iter()
        .map(|&(x, y)| rc(x, y))
        .collect()
}

/// Precomputed cache for elliptic integrals
#[derive(Clone)]
pub struct EllipticCache {
    cache: Option<Vec<f64>>,
    args: Vec<(f64, f64)>,
}

impl EllipticCache {
    /// Creates a new cache for the specified arguments
    pub fn new(args: Vec<(f64, f64)>) -> Self {
        Self {
            cache: None,
            args,
        }
    }
    
    /// Precomputes values for all arguments
    pub fn precompute(&mut self) -> EllipticResult<()> {
        let results = rc_parallel(&self.args);
        
        // Check for any errors
        if let Some(err) = results.iter().find_map(|r| r.as_ref().err()) {
            return Err(EllipticError::ComputationError(
                format!("Precomputation failed: {}", err)
            ));
        }
        
        self.cache = Some(results.into_iter().map(|r| r.unwrap()).collect());
        Ok(())
    }
    
    /// Gets a precomputed value by index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.cache.as_ref().and_then(|cache| cache.get(index).copied())
    }
    
    /// Gets all precomputed results (if available)
    pub fn results(&self) -> Option<&[f64]> {
        self.cache.as_deref()
    }
}

/// Additional utility functions

/// Computes the inverse sine function using RC
pub fn asin_rc(x: f64) -> EllipticResult<f64> {
    if x < -1.0 || x > 1.0 {
        return Err(EllipticError::InvalidArguments(
            "x must be in [-1, 1] for asin_rc".to_string()
        ));
    }
    
    rc(1.0 - x * x, 1.0).map(|result| result * x)
}

/// Computes the inverse hyperbolic sine function using RC
pub fn asinh_rc(x: f64) -> EllipticResult<f64> {
    rc(1.0 + x * x, 1.0).map(|result| result * x)
}

/// Computes the inverse cosine function using RC
pub fn acos_rc(x: f64) -> EllipticResult<f64> {
    if x < -1.0 || x > 1.0 {
        return Err(EllipticError::InvalidArguments(
            "x must be in [-1, 1] for acos_rc".to_string()
        ));
    }
    
    rc(1.0 - x * x, 1.0).map(|result| std::f64::consts::FRAC_PI_2 - result * x)
}

/// Computes the inverse hyperbolic cosine function using RC
pub fn acosh_rc(x: f64) -> EllipticResult<f64> {
    if x < 1.0 {
        return Err(EllipticError::InvalidArguments(
            "x must be >= 1 for acosh_rc".to_string()
        ));
    }
    
    rc(x * x - 1.0, 1.0).map(|result| result * x)
}

/// Benchmarking structure for performance monitoring
pub struct EllipticBenchmark {
    pub times: Vec<std::time::Duration>,
}

impl EllipticBenchmark {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
        }
    }
    
    /// Benchmarks the RC computation
    pub fn benchmark_rc(&mut self, args: &[(f64, f64)], samples: usize) {
        self.times.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = rc_parallel(args);
            self.times.push(start.elapsed());
        }
    }
    
    /// Returns the average computation time
    pub fn average_time(&self) -> std::time::Duration {
        if self.times.is_empty() {
            return std::time::Duration::from_secs(0);
        }
        
        let total: std::time::Duration = self.times.iter().sum();
        total / self.times.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rc_basic() {
        // Test cases from known values
        let result = rc(1.0, 2.0).unwrap();
        assert_abs_diff_eq!(result, 0.7854, epsilon = 1e-4);
        
        let result = rc(0.5, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.1107, epsilon = 1e-4);
    }

    #[test]
    fn test_rc_special_cases() {
        // Test case when y > 0
        let result = rc(2.0, 3.0).unwrap();
        assert_abs_diff_eq!(result, 0.726, epsilon = 1e-3);
        
        // Test case when y < 0
        let result = rc(2.0, -1.0).unwrap();
        assert_abs_diff_eq!(result, 1.209, epsilon = 1e-3);
    }

    #[test]
    fn test_rc_invalid_arguments() {
        // Test negative x
        assert!(rc(-1.0, 2.0).is_err());
        
        // Test y = 0
        assert!(rc(1.0, 0.0).is_err());
        
        // Test arguments that are too small
        assert!(rc(TINY/2.0, TINY/2.0).is_err());
        
        // Test arguments that are too large
        assert!(rc(BIG*2.0, 1.0).is_err());
    }

    #[test]
    fn test_asin_rc() {
        // Test asin(0) = 0
        let result = asin_rc(0.0).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-15);
        
        // Test asin(0.5)
        let result = asin_rc(0.5).unwrap();
        assert_abs_diff_eq!(result, 0.5236, epsilon = 1e-4);
    }

    #[test]
    fn test_parallel_computation() {
        let args = vec![
            (1.0, 2.0),
            (0.5, 1.0),
            (2.0, 3.0),
            (2.0, -1.0),
        ];
        
        let results = rc_parallel(&args);
        
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_precomputation() {
        let args = vec![
            (1.0, 2.0),
            (0.5, 1.0),
        ];
        
        let mut cache = EllipticCache::new(args.clone());
        assert!(cache.precompute().is_ok());
        
        // Test cached values
        let result1 = cache.get(0).unwrap();
        let direct1 = rc(args[0].0, args[0].1).unwrap();
        assert_abs_diff_eq!(result1, direct1, epsilon = 1e-12);
        
        let result2 = cache.get(1).unwrap();
        let direct2 = rc(args[1].0, args[1].1).unwrap();
        assert_abs_diff_eq!(result2, direct2, epsilon = 1e-12);
    }
}
