use std::error::Error;
use std::fmt;
use rayon::prelude::*;
use std::sync::OnceLock;

// Constants - Replace FRAC_1_3 with direct calculation
const ERRTOL: f64 = 0.08;
const TINY: f64 = 1.5e-38;
const BIG: f64 = 3.0e37;
const C1: f64 = 1.0 / 24.0;
const C2: f64 = 0.1;
const C3: f64 = 3.0 / 44.0;
const C4: f64 = 1.0 / 14.0;
const THIRD: f64 = 1.0 / 3.0;  // Direct calculation instead of FRAC_1_3

// Remove the OnceLock since we're using a const now
// static THIRD: OnceLock<f64> = OnceLock::new();

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

/// Carlson elliptic integral of the first kind RF(x, y, z)
/// 
/// # Arguments
/// * `x`, `y`, `z` - Parameters (must be non-negative and satisfy convergence criteria)
/// 
/// # Returns
/// The value of the integral or an error
pub fn rf(x: f64, y: f64, z: f64) -> EllipticResult<f64> {
    // Input validation
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return Err(EllipticError::InvalidArguments(
            "All arguments must be non-negative".to_string()
        ));
    }
    
    if x + y < TINY || x + z < TINY || y + z < TINY {
        return Err(EllipticError::InvalidArguments(
            "Sum of any two arguments is too small".to_string()
        ));
    }
    
    if x > BIG || y > BIG || z > BIG {
        return Err(EllipticError::InvalidArguments(
            "Arguments are too large".to_string()
        ));
    }

    // Use the const THIRD directly
    let mut xt = x;
    let mut yt = y;
    let mut zt = z;

    loop {
        let sqrtx = xt.sqrt();
        let sqrty = yt.sqrt();
        let sqrtz = zt.sqrt();
        
        let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        
        let ave = THIRD * (xt + yt + zt);
        let delx = (ave - xt) / ave;
        let dely = (ave - yt) / ave;
        let delz = (ave - zt) / ave;
        
        let max_del = delx.abs().max(dely.abs()).max(delz.abs());
        
        if max_del <= ERRTOL {
            let e2 = delx * dely - delz * delz;
            let e3 = delx * dely * delz;
            
            return Ok((1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / ave.sqrt());
        }
    }
}

/// Computes RF for multiple argument sets in parallel
pub fn rf_parallel(args: &[(f64, f64, f64)]) -> Vec<EllipticResult<f64>> {
    args.par_iter()
        .map(|&(x, y, z)| rf(x, y, z))
        .collect()
}

/// Precomputed cache for elliptic integrals
#[derive(Clone)]
pub struct EllipticCache {
    cache: Option<Vec<f64>>,
    args: Vec<(f64, f64, f64)>,
}

impl EllipticCache {
    /// Creates a new cache for the specified arguments
    pub fn new(args: Vec<(f64, f64, f64)>) -> Self {
        Self {
            cache: None,
            args,
        }
    }
    
    /// Precomputes values for all arguments
    pub fn precompute(&mut self) -> EllipticResult<()> {
        let results = rf_parallel(&self.args);
        
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

/// Computes the complete elliptic integral of the first kind K(m)
pub fn complete_elliptic_k(m: f64) -> EllipticResult<f64> {
    if m < 0.0 || m >= 1.0 {
        return Err(EllipticError::InvalidArguments(
            "Parameter m must be in [0, 1)".to_string()
        ));
    }
    
    rf(0.0, 1.0 - m, 1.0)
}

/// Computes the complete elliptic integral of the second kind E(m)
pub fn complete_elliptic_e(m: f64) -> EllipticResult<f64> {
    if m < 0.0 || m >= 1.0 {
        return Err(EllipticError::InvalidArguments(
            "Parameter m must be in [0, 1)".to_string()
        ));
    }
    
    // For E(m), we need to implement the appropriate Carlson form
    // This is a placeholder - the actual implementation would use RD
    // For now, we'll use an approximation
    if m < 0.5 {
        let k = complete_elliptic_k(m)?;
        // Approximation for small m
        Ok(k * (1.0 - m / 2.0))
    } else {
        let k = complete_elliptic_k(1.0 - m)?;
        // Approximation for m close to 1
        Ok(1.0 + (1.0 - m) * (k.ln() - 0.5))
    }
}

/// Computes the incomplete elliptic integral of the first kind F(φ, m)
pub fn incomplete_elliptic_f(phi: f64, m: f64) -> EllipticResult<f64> {
    if m < 0.0 || m >= 1.0 {
        return Err(EllipticError::InvalidArguments(
            "Parameter m must be in [0, 1)".to_string()
        ));
    }
    
    if phi < 0.0 || phi > std::f64::consts::PI / 2.0 {
        return Err(EllipticError::InvalidArguments(
            "Angle φ must be in [0, π/2]".to_string()
        ));
    }
    
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    
    rf(cos_phi * cos_phi, 1.0 - m * sin_phi * sin_phi, 1.0).map(|result| result * sin_phi)
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
    
    /// Benchmarks the RF computation
    pub fn benchmark_rf(&mut self, args: &[(f64, f64, f64)], samples: usize) {
        self.times.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = rf_parallel(args);
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
    fn test_rf_basic() {
        // Test cases from known values
        let result = rf(1.0, 2.0, 3.0).unwrap();
        assert_abs_diff_eq!(result, 0.726945, epsilon = 1e-6);
        
        let result = rf(0.5, 1.0, 1.5).unwrap();
        assert_abs_diff_eq!(result, 1.028, epsilon = 1e-3);
    }

    #[test]
    fn test_rf_special_cases() {
        // Test case when two arguments are equal
        let result = rf(2.0, 2.0, 3.0).unwrap();
        assert_abs_diff_eq!(result, 0.785, epsilon = 1e-3);
        
        // Test case when all arguments are equal
        let result = rf(3.0, 3.0, 3.0).unwrap();
        assert_abs_diff_eq!(result, 0.57735, epsilon = 1e-5); // 1/√3
    }

    #[test]
    fn test_rf_invalid_arguments() {
        // Test negative arguments
        assert!(rf(-1.0, 2.0, 3.0).is_err());
        
        // Test arguments that are too small
        assert!(rf(TINY/2.0, TINY/2.0, 1.0).is_err());
        
        // Test arguments that are too large
        assert!(rf(BIG*2.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_complete_elliptic_k() {
        // Test K(0) = π/2
        let result = complete_elliptic_k(0.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::PI / 2.0, epsilon = 1e-10);
        
        // Test known value K(0.5)
        let result = complete_elliptic_k(0.5).unwrap();
        assert_abs_diff_eq!(result, 1.854074, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_computation() {
        let args = vec![
            (1.0, 2.0, 3.0),
            (0.5, 1.0, 1.5),
            (2.0, 2.0, 3.0),
            (3.0, 3.0, 3.0),
        ];
        
        let results = rf_parallel(&args);
        
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_precomputation() {
        let args = vec![
            (1.0, 2.0, 3.0),
            (0.5, 1.0, 1.5),
        ];
        
        let mut cache = EllipticCache::new(args.clone());
        assert!(cache.precompute().is_ok());
        
        // Test cached values
        let result1 = cache.get(0).unwrap();
        let direct1 = rf(args[0].0, args[0].1, args[0].2).unwrap();
        assert_abs_diff_eq!(result1, direct1, epsilon = 1e-12);
        
        let result2 = cache.get(1).unwrap();
        let direct2 = rf(args[1].0, args[1].1, args[1].2).unwrap();
        assert_abs_diff_eq!(result2, direct2, epsilon = 1e-12);
    }
}
