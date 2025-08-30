use std::f32::consts::{PI, SQRT_2};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const MAX_ITER: usize = 100;
const EPS: f32 = 3.0e-7;
const FPMIN: f32 = 1.0e-30;

// Import the incomplete gamma functions from previous modules
mod incomplete_gamma {
    use super::*;
    
    pub fn gammp(a: f32, x: f32) -> Result<f32, String> {
        if x < 0.0 || a <= 0.0 {
            return Err("Invalid arguments".to_string());
        }
        
        if x < a + 1.0 {
            gamma_series(a, x).map(|(gamser, _)| gamser)
        } else {
            gamma_continued_fraction(a, x).map(|(gammcf, _)| 1.0 - gammcf)
        }
    }
    
    pub fn gammq(a: f32, x: f32) -> Result<f32, String> {
        if x < 0.0 || a <= 0.0 {
            return Err("Invalid arguments".to_string());
        }
        
        if x < a + 1.0 {
            gamma_series(a, x).map(|(gamser, _)| 1.0 - gamser)
        } else {
            gamma_continued_fraction(a, x).map(|(gammcf, _)| gammcf)
        }
    }
    
    fn gamma_series(a: f32, x: f32) -> Result<(f32, f32), String> {
        // Implementation from gamma_series.rs
        let gln = ln_gamma(a);
        if x == 0.0 {
            return Ok((0.0, gln));
        }
        
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        
        for _ in 1..=MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            
            if del.abs() < sum.abs() * EPS {
                let exponent = -x + a * x.ln() - gln;
                return Ok((sum * exponent.exp(), gln));
            }
        }
        
        Err("Max iterations exceeded".to_string())
    }
    
    fn gamma_continued_fraction(a: f32, x: f32) -> Result<(f32, f32), String> {
        // Implementation from gamma_continued_fraction.rs
        let gln = ln_gamma(a);
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        
        for i in 1..=MAX_ITER {
            let an = -(i as f32) * (i as f32 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN { d = FPMIN; }
            
            c = b + an / c;
            if c.abs() < FPMIN { c = FPMIN; }
            
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            
            if (del - 1.0).abs() < EPS {
                let exponent = -x + a * x.ln() - gln;
                return Ok((exponent.exp() * h, gln));
            }
        }
        
        Err("Max iterations exceeded".to_string())
    }
}

/// Calculates the natural logarithm of the gamma function
fn ln_gamma(x: f32) -> f32 {
    const COEFFS: [f32; 7] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
    ];
    
    if x <= 0.0 { return f32::NAN; }
    if x < 0.5 {
        return (PI / ((PI * x).sin())).ln() - ln_gamma(1.0 - x);
    }
    
    let mut x = x - 1.0;
    let mut t = COEFFS[0];
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        t += coeff / (x + i as f32);
    }
    
    let sqrt_2pi = (2.0 * PI).sqrt();
    (sqrt_2pi * t).ln() + (x + 0.5) * (x + 1.5 + 5.5).ln() - (x + 1.5 + 5.5)
}

/// Error function using incomplete gamma function
/// erf(x) = 2/√π ∫₀ˣ e^(-t²) dt
pub fn erf(x: f32) -> Result<f32, String> {
    let x_sq = x * x;
    let result = if x < 0.0 {
        -incomplete_gamma::gammp(0.5, x_sq)?
    } else {
        incomplete_gamma::gammp(0.5, x_sq)?
    };
    Ok(result)
}

/// Complementary error function using incomplete gamma function
/// erfc(x) = 1 - erf(x) = 2/√π ∫ₓ∞ e^(-t²) dt
pub fn erfc_gamma(x: f32) -> Result<f32, String> {
    let x_sq = x * x;
    let result = if x < 0.0 {
        1.0 + incomplete_gamma::gammp(0.5, x_sq)?
    } else {
        incomplete_gamma::gammq(0.5, x_sq)?
    };
    Ok(result)
}

/// Complementary error function using Chebyshev approximation
/// More accurate for large |x| values
pub fn erfc_chebyshev(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    
    // Chebyshev polynomial coefficients
    let ans = t * (-z * z - 1.26551223 + t * (
        1.000002368 + t * (
            0.37409196 + t * (
                0.09678418 + t * (
                    -0.18628806 + t * (
                        0.27886807 + t * (
                            -1.13520398 + t * (
                                1.48851587 + t * (
                                    -0.82215223 + t * 0.17087277
                                )
                            )
                        )
                    )
                )
            )
        )
    )).exp();
    
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Error function using Chebyshev approximation
pub fn erf_chebyshev(x: f32) -> f32 {
    1.0 - erfc_chebyshev(x)
}

/// Smart error function that chooses the best algorithm automatically
pub fn erf_smart(x: f32) -> f32 {
    if x.abs() < 3.0 {
        // Use gamma-based method for small |x| (more accurate near 0)
        erf(x).unwrap_or_else(|_| erf_chebyshev(x))
    } else {
        // Use Chebyshev for large |x| (more stable)
        erf_chebyshev(x)
    }
}

/// Smart complementary error function that chooses the best algorithm
pub fn erfc_smart(x: f32) -> f32 {
    if x.abs() < 3.0 {
        // Use gamma-based method for small |x|
        erfc_gamma(x).unwrap_or_else(|_| erfc_chebyshev(x))
    } else {
        // Use Chebyshev for large |x|
        erfc_chebyshev(x)
    }
}

/// Thread-safe cache for error function values
#[derive(Clone)]
pub struct ErrorFunctionCache {
    erf_cache: Arc<Mutex<std::collections::HashMap<i32, f32>>>,
    erfc_cache: Arc<Mutex<std::collections::HashMap<i32, f32>>>,
}

impl ErrorFunctionCache {
    pub fn new() -> Self {
        Self {
            erf_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            erfc_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get erf value with caching (quantized input)
    pub fn erf_cached(&self, x: f32) -> f32 {
        let key = (x * 1000.0) as i32;
        let mut cache = self.erf_cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return result;
        }
        
        let result = erf_smart(x);
        cache.insert(key, result);
        result
    }
    
    /// Get erfc value with caching (quantized input)
    pub fn erfc_cached(&self, x: f32) -> f32 {
        let key = (x * 1000.0) as i32;
        let mut cache = self.erfc_cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return result;
        }
        
        let result = erfc_smart(x);
        cache.insert(key, result);
        result
    }
    
    pub fn clear(&self) {
        self.erf_cache.lock().unwrap().clear();
        self.erfc_cache.lock().unwrap().clear();
    }
}

/// Parallel batch computation of error functions
pub fn erf_batch_parallel(x_values: &[f32]) -> Vec<f32> {
    x_values.par_iter()
        .map(|&x| erf_smart(x))
        .collect()
}

pub fn erfc_batch_parallel(x_values: &[f32]) -> Vec<f32> {
    x_values.par_iter()
        .map(|&x| erfc_smart(x))
        .collect()
}

/// Compute both erf and erfc efficiently
pub fn both_error_functions(x: f32) -> (f32, f32) {
    let erfc = erfc_smart(x);
    (1.0 - erfc, erfc)
}

/// Compute both erf and erfc in parallel for multiple values
pub fn both_error_functions_batch_parallel(x_values: &[f32]) -> Vec<(f32, f32)> {
    x_values.par_iter()
        .map(|&x| both_error_functions(x))
        .collect()
}

/// Inverse error function approximation
pub fn erf_inv(y: f32) -> Result<f32, String> {
    if y <= -1.0 || y >= 1.0 {
        return Err("y must be in (-1, 1)".to_string());
    }
    
    // Use Newton's method with erf_chebyshev as approximation
    let mut x = 0.0;
    let mut delta;
    
    for _ in 0..10 {
        let erf_x = erf_chebyshev(x);
        let derivative = (2.0 / PI.sqrt()) * (-x * x).exp();
        
        if derivative.abs() < 1e-10 {
            break;
        }
        
        delta = (erf_x - y) / derivative;
        x -= delta;
        
        if delta.abs() < 1e-10 {
            break;
        }
    }
    
    Ok(x)
}

/// Inverse complementary error function
pub fn erfc_inv(y: f32) -> Result<f32, String> {
    if y <= 0.0 || y >= 2.0 {
        return Err("y must be in (0, 2)".to_string());
    }
    erf_inv(1.0 - y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_erf_basic() {
        // Test known values
        assert_relative_eq!(erf(0.0).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(erf(1.0).unwrap(), 0.84270079, epsilon = 1e-6);
        assert_relative_eq!(erf(-1.0).unwrap(), -0.84270079, epsilon = 1e-6);
    }

    #[test]
    fn test_erfc_basic() {
        assert_relative_eq!(erfc_gamma(0.0).unwrap(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(erfc_gamma(1.0).unwrap(), 0.15729921, epsilon = 1e-6);
        assert_relative_eq!(erfc_gamma(-1.0).unwrap(), 1.84270079, epsilon = 1e-6);
    }

    #[test]
    fn test_erfc_chebyshev() {
        // Test Chebyshev approximation
        assert_relative_eq!(erfc_chebyshev(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(erfc_chebyshev(1.0), 0.15729921, epsilon = 1e-6);
        assert_relative_eq!(erfc_chebyshev(3.0), 0.00002209, epsilon = 1e-6);
    }

    #[test]
    fn test_erf_erfc_relationship() {
        // Test that erf(x) + erfc(x) = 1
        let test_cases = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        
        for &x in &test_cases {
            let erf_val = erf_smart(x);
            let erfc_val = erfc_smart(x);
            assert_relative_eq!(erf_val + erfc_val, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_erf_symmetry() {
        // Test that erf(-x) = -erf(x)
        for x in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
            assert_relative_eq!(erf_smart(-x), -erf_smart(x), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_algorithm_consistency() {
        // Test that different algorithms give consistent results
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let erf_gamma = erf(x).unwrap_or(0.0);
            let erf_cheb = erf_chebyshev(x);
            let erf_smart_val = erf_smart(x);
            
            if x.abs() < 3.0 {
                assert_relative_eq!(erf_gamma, erf_cheb, epsilon = 1e-4);
            }
            assert_relative_eq!(erf_smart_val, erf_cheb, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
        
        let erf_results = erf_batch_parallel(&x_values);
        let erfc_results = erfc_batch_parallel(&x_values);
        
        for i in 0..x_values.len() {
            assert_relative_eq!(erf_results[i] + erfc_results[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_both_error_functions() {
        let (erf_val, erfc_val) = both_error_functions(1.0);
        assert_relative_eq!(erf_val, 0.84270079, epsilon = 1e-6);
        assert_relative_eq!(erfc_val, 0.15729921, epsilon = 1e-6);
        assert_relative_eq!(erf_val + erfc_val, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cache_functionality() {
        let cache = ErrorFunctionCache::new();
        
        let result1 = cache.erf_cached(1.0);
        let result2 = cache.erf_cached(1.0); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, erf_smart(1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_error_functions() {
        // Test inverse functions
        for x in [-1.5, -0.5, 0.5, 1.5] {
            let y = erf_smart(x);
            let x_inv = erf_inv(y).unwrap();
            assert_relative_eq!(x, x_inv, epsilon = 1e-4);
            
            let y_erfc = erfc_smart(x);
            let x_erfc_inv = erfc_inv(y_erfc).unwrap();
            assert_relative_eq!(x, x_erfc_inv, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_inverse_error_function_edges() {
        // Test edge cases for inverse functions
        assert!(erf_inv(-1.1).is_err());
        assert!(erf_inv(1.1).is_err());
        assert!(erfc_inv(-0.1).is_err());
        assert!(erfc_inv(2.1).is_err());
    }

    #[test]
    fn test_large_values() {
        // Test behavior for large values
        assert_relative_eq!(erf_smart(10.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(erfc_smart(10.0), 0.0, epsilon = 1e-10);
        
        assert_relative_eq!(erf_smart(-10.0), -1.0, epsilon = 1e-10);
        assert_relative_eq!(erfc_smart(-10.0), 2.0, epsilon = 1e-10);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_erf_smart(c: &mut Criterion) {
        c.bench_function("erf_smart(1.0)", |b| {
            b.iter(|| erf_smart(black_box(1.0)))
        });
        
        c.bench_function("erf_smart(3.0)", |b| {
            b.iter(|| erf_smart(black_box(3.0)))
        });
    }

    fn bench_erfc_smart(c: &mut Criterion) {
        c.bench_function("erfc_smart(1.0)", |b| {
            b.iter(|| erfc_smart(black_box(1.0)))
        });
        
        c.bench_function("erfc_smart(3.0)", |b| {
            b.iter(|| erfc_smart(black_box(3.0)))
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let x_values: Vec<f32> = (-100..100).map(|x| x as f32 / 10.0).collect();
        
        c.bench_function("erf_batch_parallel(200)", |b| {
            b.iter(|| erf_batch_parallel(black_box(&x_values)))
        });
        
        c.bench_function("both_error_functions_batch_parallel(200)", |b| {
            b.iter(|| both_error_functions_batch_parallel(black_box(&x_values)))
        });
    }

    fn bench_cache(c: &mut Criterion) {
        let cache = ErrorFunctionCache::new();
        
        c.bench_function("erf_cached(1.0)", |b| {
            b.iter(|| cache.erf_cached(black_box(1.0)))
        });
    }

    criterion_group!(
        benches, 
        bench_erf_smart, 
        bench_erfc_smart, 
        bench_batch_parallel,
        bench_cache
    );
    criterion_main!(benches);
}
