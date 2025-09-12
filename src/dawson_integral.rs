use std::error::Error;
use std::fmt;
use std::f64::consts::{PI, SQRT_2};
use rayon::prelude::*;
use once_cell::sync::Lazy;

// Constants
const NMAX: usize = 6;
const H: f64 = 0.4;
const A1: f64 = 2.0 / 3.0;
const A2: f64 = 0.4;
const A3: f64 = 2.0 / 7.0;

// Precomputed constants
static C: Lazy<[f64; NMAX + 1]> = Lazy::new(|| {
    let mut c = [0.0; NMAX + 1];
    for i in 1..=NMAX {
        let term = (2.0 * i as f64 - 1.0) * H;
        c[i] = (term * term).exp();
    }
    c
});

// Custom error type
#[derive(Debug, Clone)]
pub enum DawsonError {
    ComputationError(String),
}

impl fmt::Display for DawsonError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DawsonError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for DawsonError {}

// Result type alias
pub type DawsonResult<T> = Result<T, DawsonError>;

/// Computes Dawson's integral F(x) = exp(-x²) ∫₀ˣ exp(t²) dt
pub fn dawson(x: f64) -> f64 {
    if x.abs() < 0.2 {
        // Small x approximation
        let x2 = x * x;
        x * (1.0 - A1 * x2 * (1.0 - A2 * x2 * (1.0 - A3 * x2)))
    } else {
        // Large x using series expansion
        let xx = x.abs();
        let n0 = (0.5 * xx / H + 0.5).round() as i32 * 2;
        let xp = xx - n0 as f64 * H;
        
        let mut e1 = (2.0 * xp * H).exp();
        let e2 = e1 * e1;
        
        let mut d1 = n0 as f64 + 1.0;
        let mut d2 = d1 - 2.0;
        
        let mut sum = 0.0;
        
        for i in 1..=NMAX {
            sum += C[i] * (e1 / d1 + 1.0 / (d2 * e1));
            d1 += 2.0;
            d2 -= 2.0;
            e1 *= e2;
        }
        
        // SIGN equivalent: if x >= 0 use positive, else negative
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        0.5641895835 * sign * (-xp * xp).exp() * sum
    }
}

/// Computes Dawson's integral for multiple values in parallel
pub fn dawson_parallel(x_values: &[f64]) -> Vec<f64> {
    x_values.par_iter().map(|&x| dawson(x)).collect()
}

/// Computes the scaled Dawson function F(x) * exp(x²)
pub fn dawson_scaled(x: f64) -> f64 {
    let dawson_val = dawson(x);
    dawson_val * (x * x).exp()
}

/// Precomputed cache for Dawson's integral
#[derive(Clone)]
pub struct DawsonComputer {
    cache: Option<Vec<f64>>,
    x_min: f64,
    x_max: f64,
    step: f64,
    x_values: Vec<f64>,
}

impl DawsonComputer {
    /// Creates a new Dawson computer
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
    pub fn precompute(&mut self) {
        self.cache = Some(dawson_parallel(&self.x_values));
    }
    
    /// Gets Dawson's integral for a specific x value
    pub fn get(&self, x: f64) -> f64 {
        // Use cache for values within the precomputed range
        if let Some(cache) = &self.cache {
            if x >= self.x_min && x <= self.x_max {
                let idx = ((x - self.x_min) / self.step).round() as usize;
                if idx < cache.len() {
                    return cache[idx];
                }
            }
        }
        
        // Fall back to direct computation
        dawson(x)
    }
    
    /// Gets all precomputed x values
    pub fn x_values(&self) -> &[f64] {
        &self.x_values
    }
    
    /// Gets all precomputed results (if available)
    pub fn results(&self) -> Option<&[f64]> {
        self.cache.as_deref()
    }
}

/// Additional utility functions

/// Computes the error function using Dawson's integral
pub fn erf_via_dawson(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    
    let dawson_val = dawson(x);
    2.0 / PI.sqrt() * (x * x).exp() * dawson_val
}

/// Computes the complementary error function
pub fn erfc_via_dawson(x: f64) -> f64 {
    1.0 - erf_via_dawson(x)
}

/// Computes the imaginary error function erfi(x) = -i * erf(ix)
pub fn erfi_via_dawson(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    
    let dawson_val = dawson(-x);
    2.0 / PI.sqrt() * (x * x).exp() * dawson_val
}

/// Computes the Faddeeva function w(z) = exp(-z²) * erfc(-iz)
pub fn faddeeva(z: f64) -> f64 {
    if z == 0.0 {
        return 1.0;
    }
    
    let z_sq = z * z;
    let exp_term = (-z_sq).exp();
    let dawson_term = dawson(z);
    
    exp_term + 2.0 * z / PI.sqrt() * dawson_term
}

/// Benchmarking structure for performance monitoring
pub struct DawsonBenchmark {
    pub times: Vec<std::time::Duration>,
}

impl DawsonBenchmark {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
        }
    }
    
    /// Benchmarks the Dawson computation
    pub fn benchmark(&mut self, x_values: &[f64], samples: usize) {
        self.times.clear();
        
        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = dawson_parallel(x_values);
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
    fn test_small_values() {
        // Test very small x
        assert_abs_diff_eq!(dawson(0.0), 0.0, epsilon = 1e-15);
        
        // Test small positive x
        assert_abs_diff_eq!(dawson(0.1), 0.099335, epsilon = 1e-6);
        assert_abs_diff_eq!(dawson(0.2), 0.194751, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_values() {
        // Test symmetry: Dawson's integral is odd
        let pos = dawson(1.0);
        let neg = dawson(-1.0);
        assert_abs_diff_eq!(pos, -neg, epsilon = 1e-15);
        
        // Test specific negative value
        assert_abs_diff_eq!(dawson(-0.5), -0.424436, epsilon = 1e-6);
    }

    #[test]
    fn test_large_values() {
        // Test large positive x
        assert_abs_diff_eq!(dawson(2.0), 0.301340, epsilon = 1e-6);
        assert_abs_diff_eq!(dawson(5.0), 0.102134, epsilon = 1e-6);
        
        // Test very large x (should approach 1/(2x))
        let large_x = 10.0;
        let expected = 1.0 / (2.0 * large_x);
        assert_abs_diff_eq!(dawson(large_x), expected, epsilon = 1e-3);
    }

    #[test]
    fn test_parallel_computation() {
        let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let results = dawson_parallel(&x_values);
        
        assert_eq!(results.len(), 5);
        
        // Verify symmetry
        assert_abs_diff_eq!(results[0], -results[4], epsilon = 1e-15);
        assert_abs_diff_eq!(results[1], -results[3], epsilon = 1e-15);
        assert_abs_diff_eq!(results[2], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_precomputation() {
        let mut computer = DawsonComputer::new(-2.0, 2.0, 100);
        computer.precompute();
        
        // Test cached values
        let result = computer.get(1.0);
        let direct = dawson(1.0);
        
        assert_abs_diff_eq!(result, direct, epsilon = 1e-12);
    }

    #[test]
    fn test_related_functions() {
        // Test error function via Dawson
        let x = 1.0;
        let erf_val = erf_via_dawson(x);
        assert_abs_diff_eq!(erf_val, 0.842700, epsilon = 1e-6);
        
        // Test Faddeeva function
        let faddeeva_val = faddeeva(x);
        assert_abs_diff_eq!(faddeeva_val, 0.427584, epsilon = 1e-6);
    }
}
