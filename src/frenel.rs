use std::error::Error;
use std::fmt;
use std::f64::consts::PI;
use rayon::prelude::*;
use num_complex::Complex64;

// Constants
const EPS: f64 = 6.0e-8;
const MAXIT: usize = 100;
const FPMIN: f64 = 1.0e-30;
const XMIN: f64 = 1.5;
const PIBY2: f64 = PI / 2.0;

// Custom error type
#[derive(Debug, Clone)]
pub enum FresnelError {
    ComputationError(String),
    ConvergenceError(String),
}

impl fmt::Display for FresnelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FresnelError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            FresnelError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
        }
    }
}

impl Error for FresnelError {}

// Result type alias
pub type FresnelResult<T> = Result<T, FresnelError>;

/// Fresnel integrals S(x) and C(x)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FresnelIntegrals {
    pub s: f64,
    pub c: f64,
}

impl FresnelIntegrals {
    pub fn new(s: f64, c: f64) -> Self {
        Self { s, c }
    }
}

/// Computes Fresnel integrals S(x) and C(x) for a single value
pub fn frenel(x: f64) -> FresnelResult<FresnelIntegrals> {
    let ax = x.abs();
    
    // Handle very small x values
    if ax < FPMIN.sqrt() {
        return Ok(FresnelIntegrals::new(0.0, ax));
    }
    
    let (s, c) = if ax <= XMIN {
        // Series expansion for small x
        frenel_series(ax)?
    } else {
        // Continued fraction for large x
        frenel_continued_fraction(ax)?
    };
    
    // Adjust sign for negative x
    if x < 0.0 {
        Ok(FresnelIntegrals::new(-s, -c))
    } else {
        Ok(FresnelIntegrals::new(s, c))
    }
}

/// Series expansion for Fresnel integrals (small x)
fn frenel_series(ax: f64) -> FresnelResult<(f64, f64)> {
    let mut sum = 0.0;
    let mut sums = 0.0;
    let mut sumc = ax;
    let mut sign = 1.0;
    let fact = PIBY2 * ax * ax;
    let mut odd = true;
    let mut term = ax;
    let mut n = 3;

    for k in 1..=MAXIT {
        term *= fact / k as f64;
        sum += sign * term / n as f64;
        
        let test = sum.abs() * EPS;
        
        if odd {
            sign = -sign;
            sums = sum;
            sum = sumc;
        } else {
            sumc = sum;
            sum = sums;
        }
        
        if term < test {
            return Ok((sums, sumc));
        }
        
        odd = !odd;
        n += 2;
    }
    
    Err(FresnelError::ConvergenceError(
        "Series expansion did not converge in frenel_series".to_string()
    ))
}

/// Continued fraction method for Fresnel integrals (large x)
fn frenel_continued_fraction(ax: f64) -> FresnelResult<(f64, f64)> {
    let pix2 = PI * ax * ax;
    let mut b = Complex64::new(1.0, -pix2);
    let mut cc = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = Complex64::new(1.0, 0.0) / b;
    let mut h = d;
    let mut n: i32 = -1;

    for k in 2..=MAXIT {
        n += 2;
        let a = -(n * (n + 1)) as f64;
        
        b += Complex64::new(4.0, 0.0);
        
        d = Complex64::new(1.0, 0.0) / (a * d + b);
        cc = b + Complex64::new(a, 0.0) / cc;
        
        let del = cc * d;
        h *= del;
        
        if (del.re - 1.0).abs() + del.im.abs() < EPS {
            h *= Complex64::new(ax, -ax);
            
            let phase = Complex64::new((0.5 * pix2).cos(), (0.5 * pix2).sin());
            let cs = Complex64::new(0.5, 0.5) * (Complex64::new(1.0, 0.0) - phase * h);
            
            return Ok((cs.im, cs.re));
        }
    }
    
    Err(FresnelError::ConvergenceError(
        "Continued fraction did not converge in frenel_continued_fraction".to_string()
    ))
}

/// Computes Fresnel integrals for multiple values in parallel
pub fn frenel_parallel(x_values: &[f64]) -> Vec<FresnelResult<FresnelIntegrals>> {
    x_values
        .par_iter()
        .map(|&x| frenel(x))
        .collect()
}

/// Computes both Fresnel integrals for a range of x values with caching
#[derive(Clone)]
pub struct FresnelComputer {
    cache: Option<Vec<FresnelIntegrals>>,
    x_min: f64,
    x_max: f64,
    step: f64,
    x_values: Vec<f64>,
}

impl FresnelComputer {
    /// Creates a new Fresnel computer with optional precomputation
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
    pub fn precompute(&mut self) -> FresnelResult<()> {
        let results = frenel_parallel(&self.x_values);
        
        // Check for any errors
        if let Some(err) = results.iter().find_map(|r| r.as_ref().err()) {
            return Err(FresnelError::ComputationError(
                format!("Precomputation failed: {}", err)
            ));
        }
        
        self.cache = Some(results.into_iter().map(|r| r.unwrap()).collect());
        Ok(())
    }
    
    /// Gets Fresnel integrals for a specific x value, using cache if available
    pub fn get(&self, x: f64) -> FresnelResult<FresnelIntegrals> {
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
        frenel(x)
    }
    
    /// Gets all precomputed x values
    pub fn x_values(&self) -> &[f64] {
        &self.x_values
    }
    
    /// Gets all precomputed results (if available)
    pub fn results(&self) -> Option<&[FresnelIntegrals]> {
        self.cache.as_deref()
    }
}

/// Additional utility functions

/// Computes the Cornu spiral (parametric curve of Fresnel integrals)
pub fn cornu_spiral(t_values: &[f64]) -> Vec<(f64, f64)> {
    t_values
        .par_iter()
        .map(|&t| {
            let result = frenel(t).unwrap_or_else(|_| FresnelIntegrals::new(0.0, 0.0));
            (result.c, result.s)
        })
        .collect()
}

/// Computes the Fresnel integrals using asymptotic expansion for very large x
pub fn frenel_asymptotic(x: f64) -> FresnelResult<FresnelIntegrals> {
    if x.abs() < XMIN {
        return frenel(x);
    }
    
    let ax = x.abs();
    let pix2 = PI * ax * ax;
    
    // Asymptotic expansion terms
    let term1 = 0.5 + (0.5 * pix2).sin() / (PI * ax);
    let term2 = - (0.5 * pix2).cos() / (PI * ax);
    
    let c = if x > 0.0 { term1 } else { 1.0 - term1 };
    let s = if x > 0.0 { term2 } else { 0.5 - term2 };
    
    Ok(FresnelIntegrals::new(s, c))
}

/// Benchmarking and performance monitoring
pub struct FresnelBenchmark {
    pub times: Vec<std::time::Duration>,
    pub errors: Vec<F64>,
}

impl FresnelBenchmark {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            errors: Vec::new(),
        }
    }
    
    /// Benchmarks the Fresnel computation for a set of values
    pub fn benchmark(&mut self, x_values: &[f64], samples: usize) {
        self.times.clear();
        self.errors.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let results: Vec<_> = x_values
                .par_iter()
                .map(|&x| frenel(x))
                .collect();
            let duration = start.elapsed();
            
            self.times.push(duration);
            
            // Calculate average error (placeholder)
            let avg_error = results.iter()
                .filter_map(|r| r.as_ref().err().map(|_| 1.0))
                .sum::<f64>() / x_values.len() as f64;
            
            self.errors.push(avg_error);
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
        let result = frenel(1e-10).unwrap();
        assert_abs_diff_eq!(result.s, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result.c, 1e-10, epsilon = 1e-15);
        
        // Test small positive x
        let result = frenel(0.5).unwrap();
        assert_abs_diff_eq!(result.s, 0.064732432, epsilon = 1e-8);
        assert_abs_diff_eq!(result.c, 0.492344, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_values() {
        let pos = frenel(1.0).unwrap();
        let neg = frenel(-1.0).unwrap();
        
        assert_abs_diff_eq!(pos.s, -neg.s, epsilon = 1e-15);
        assert_abs_diff_eq!(pos.c, -neg.c, epsilon = 1e-15);
    }

    #[test]
    fn test_large_values() {
        // Test large positive x
        let result = frenel(5.0).unwrap();
        assert_abs_diff_eq!(result.s, 0.563631, epsilon = 1e-6);
        assert_abs_diff_eq!(result.c, 0.499191, epsilon = 1e-6);
        
        // Test very large x (should approach 0.5)
        let result = frenel(100.0).unwrap();
        assert_abs_diff_eq!(result.s, 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.c, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_parallel_computation() {
        let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let results = frenel_parallel(&x_values);
        
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));
        
        // Verify symmetry
        let pos_1 = results[3].as_ref().unwrap();
        let neg_1 = results[1].as_ref().unwrap();
        assert_abs_diff_eq!(pos_1.s, -neg_1.s, epsilon = 1e-15);
        assert_abs_diff_eq!(pos_1.c, -neg_1.c, epsilon = 1e-15);
    }

    #[test]
    fn test_precomputation() {
        let mut computer = FresnelComputer::new(-2.0, 2.0, 100);
        assert!(computer.precompute().is_ok());
        
        // Test cached values
        let result = computer.get(1.0).unwrap();
        let direct = frenel(1.0).unwrap();
        
        assert_abs_diff_eq!(result.s, direct.s, epsilon = 1e-12);
        assert_abs_diff_eq!(result.c, direct.c, epsilon = 1e-12);
    }
}
