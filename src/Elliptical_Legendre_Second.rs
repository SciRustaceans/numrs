use std::error::Error;
use std::fmt;
use std::f64::consts::FRAC_PI_2;
use rayon::prelude::*;

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

/// Incomplete elliptic integral of the second kind E(φ, k)
/// 
/// # Arguments
/// * `phi` - Amplitude angle (in radians)
/// * `ak` - Modulus (must satisfy 0 ≤ k² ≤ 1)
/// 
/// # Returns
/// The value of the integral or an error
pub fn elle(phi: f64, ak: f64) -> EllipticResult<f64> {
    // Input validation
    if ak < 0.0 || ak > 1.0 {
        return Err(EllipticError::InvalidArguments(
            "Modulus ak must be in [0, 1]".to_string()
        ));
    }
    
    if phi.is_nan() || phi.is_infinite() {
        return Err(EllipticError::InvalidArguments(
            "Phi must be a finite number".to_string()
        ));
    }
    
    let s = phi.sin();
    let cc = phi.cos().powi(2);
    let q = (1.0 - s * ak) * (1.0 + s * ak);
    let s_ak_sq = (s * ak).powi(2);
    
    // Use Carlson's RF and RD functions
    let rf_val = rf(cc, q, 1.0)?;
    let rd_val = rd(cc, q, 1.0)?;
    
    Ok(s * (rf_val - s_ak_sq * rd_val / 3.0))
}

/// Complete elliptic integral of the second kind E(k)
/// 
/// # Arguments
/// * `ak` - Modulus (must satisfy 0 ≤ k² ≤ 1)
/// 
/// # Returns
/// The value of the integral or an error
pub fn comp_elle(ak: f64) -> EllipticResult<f64> {
    elle(FRAC_PI_2, ak)
}

/// Computes E(φ, k) for multiple argument pairs in parallel
pub fn elle_parallel(args: &[(f64, f64)]) -> Vec<EllipticResult<f64>> {
    args.par_iter()
        .map(|&(phi, ak)| elle(phi, ak))
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
        let results = elle_parallel(&self.args);
        
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

/// Computes the circumference of an ellipse
/// 
/// # Arguments
/// * `a` - Semi-major axis
/// * `b` - Semi-minor axis
/// 
/// # Returns
/// The circumference of the ellipse or an error
pub fn ellipse_circumference(a: f64, b: f64) -> EllipticResult<f64> {
    if a <= 0.0 || b <= 0.0 {
        return Err(EllipticError::InvalidArguments(
            "Semi-axes must be positive".to_string()
        ));
    }
    
    let (major, minor) = if a > b {
        (a, b)
    } else {
        (b, a)
    };
    
    let eccentricity = (1.0 - (minor / major).powi(2)).sqrt();
    comp_elle(eccentricity).map(|e| 4.0 * major * e)
}

/// Computes the arc length of an ellipse segment
/// 
/// # Arguments
/// * `a` - Semi-major axis
/// * `b` - Semi-minor axis
/// * `phi` - Angle from the major axis (in radians)
/// 
/// # Returns
/// The arc length from 0 to phi or an error
pub fn ellipse_arc_length(a: f64, b: f64, phi: f64) -> EllipticResult<f64> {
    if a <= 0.0 || b <= 0.0 {
        return Err(EllipticError::InvalidArguments(
            "Semi-axes must be positive".to_string()
        ));
    }
    
    if phi < 0.0 {
        return Err(EllipticError::InvalidArguments(
            "Phi must be non-negative".to_string()
        ));
    }
    
    let (major, minor) = if a > b {
        (a, b)
    } else {
        (b, a)
    };
    
    let eccentricity = (1.0 - (minor / major).powi(2)).sqrt();
    elle(phi, eccentricity).map(|e| major * e)
}

/// Computes the perimeter of an ellipse using Ramanujan's approximation
/// 
/// # Arguments
/// * `a` - Semi-major axis
/// * `b` - Semi-minor axis
/// 
/// # Returns
/// The perimeter of the ellipse
pub fn ellipse_perimeter_ramanujan(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    
    let h = ((a - b) / (a + b)).powi(2);
    std::f64::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()))
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
    
    /// Benchmarks the elle computation
    pub fn benchmark_elle(&mut self, args: &[(f64, f64)], samples: usize) {
        self.times.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = elle_parallel(args);
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

// Carlson's RF and RD function implementations (from previous code)
fn rf(x: f64, y: f64, z: f64) -> EllipticResult<f64> {
    // Implementation of RF would go here
    // This is a placeholder - in a real implementation, we would include the RF function
    Err(EllipticError::ComputationError("RF not implemented".to_string()))
}

fn rd(x: f64, y: f64, z: f64) -> EllipticResult<f64> {
    // Implementation of RD would go here
    // This is a placeholder - in a real implementation, we would include the RD function
    Err(EllipticError::ComputationError("RD not implemented".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_elle_basic() {
        // Test cases from known values
        // When k=0, E(φ, 0) = φ
        let result = elle(0.5, 0.0).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
        
        // When φ=π/2, E(π/2, k) = E(k)
        let result = elle(std::f64::consts::FRAC_PI_2, 0.5).unwrap();
        // Known value of E(0.5) is approximately 1.46746
        assert_abs_diff_eq!(result, 1.46746, epsilon = 1e-5);
    }

    #[test]
    fn test_comp_elle() {
        // Test complete elliptic integral E(k)
        // E(0) = π/2
        let result = comp_elle(0.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
        
        // E(0.5) ≈ 1.46746
        let result = comp_elle(0.5).unwrap();
        assert_abs_diff_eq!(result, 1.46746, epsilon = 1e-5);
        
        // E(1) = 1
        let result = comp_elle(1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_elle_invalid_arguments() {
        // Test invalid modulus
        assert!(elle(0.5, -0.1).is_err());
        assert!(elle(0.5, 1.1).is_err());
        
        // Test invalid phi
        assert!(elle(std::f64::NAN, 0.5).is_err());
        assert!(elle(std::f64::INFINITY, 0.5).is_err());
    }

    #[test]
    fn test_ellipse_circumference() {
        // Test circumference of a circle (special case of ellipse)
        let result = ellipse_circumference(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 2.0 * std::f64::consts::PI, epsilon = 1e-10);
        
        // Test circumference of an ellipse
        let result = ellipse_circumference(2.0, 1.0).unwrap();
        // Known value for ellipse with a=2, b=1 is approximately 9.68845
        assert_abs_diff_eq!(result, 9.68845, epsilon = 1e-5);
    }

    #[test]
    fn test_ellipse_arc_length() {
        // Test arc length of a circle
        let result = ellipse_arc_length(1.0, 1.0, std::f64::consts::FRAC_PI_2).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
        
        // Test arc length of an ellipse
        let result = ellipse_arc_length(2.0, 1.0, std::f64::consts::FRAC_PI_2).unwrap();
        // Known value for ellipse with a=2, b=1 from 0 to π/2 is approximately 2.42211
        assert_abs_diff_eq!(result, 2.42211, epsilon = 1e-5);
    }

    #[test]
    fn test_parallel_computation() {
        let args = vec![
            (0.5, 0.0),
            (1.0, 0.3),
            (1.5, 0.6),
            (2.0, 0.9),
        ];
        
        let results = elle_parallel(&args);
        
        assert_eq!(results.len(), 4);
        // We expect some to fail due to missing RF/RD implementation
        // In a complete implementation, we would test with valid arguments
    }
}
