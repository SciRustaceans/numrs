use std::f32::consts::{E, PI};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const MAX_ITER: usize = 100;
const EPS: f32 = 3.0e-7;
const FPMIN: f32 = 1.0e-30;

/// Calculates the natural logarithm of the gamma function using Lanczos approximation
pub fn ln_gamma(x: f32) -> f32 {
    // Lanczos approximation coefficients for f32 precision
    const COEFFS: [f32; 7] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
    ];
    
    if x <= 0.0 {
        return f32::NAN;
    }
    
    // Reflection formula for x < 0.5
    if x < 0.5 {
        let pi = PI;
        return (pi / ((pi * x).sin())).ln() - ln_gamma(1.0 - x);
    }
    
    let mut x = x - 1.0;
    let mut t = COEFFS[0];
    
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        t += coeff / (x + i as f32);
    }
    
    let sqrt_2pi = (2.0 * PI).sqrt();
    (sqrt_2pi * t).ln() + (x + 0.5) * (x + 1.5 + 5.5).ln() - (x + 1.5 + 5.5)
}

/// Regularized lower incomplete gamma function P(a, x)
/// Computes γ(a, x) / Γ(a) using series representation
pub fn gammp(a: f32, x: f32) -> Result<f32, String> {
    if x < 0.0 || a <= 0.0 {
        return Err("Invalid arguments: x must be >= 0 and a must be > 0".to_string());
    }
    
    if x < a + 1.0 {
        let (gammser, _) = gser(a, x)?;
        Ok(gammser)
    } else {
        let (gammcf, _) = gcf(a, x)?;
        Ok(1.0 - gammcf)
    }
}

/// Regularized upper incomplete gamma function Q(a, x)
/// Computes Γ(a, x) / Γ(a) using continued fraction representation
pub fn gammq(a: f32, x: f32) -> Result<f32, String> {
    if x < 0.0 || a <= 0.0 {
        return Err("Invalid arguments: x must be >= 0 and a must be > 0".to_string());
    }
    
    if x < a + 1.0 {
        let (gammser, _) = gser(a, x)?;
        Ok(1.0 - gammser)
    } else {
        let (gammcf, _) = gcf(a, x)?;
        Ok(gammcf)
    }
}

/// Series representation for incomplete gamma function
/// Returns (result, ln Γ(a))
fn gser(a: f32, x: f32) -> Result<(f32, f32), String> {
    if x < 0.0 || a <= 0.0 {
        return Err("Invalid arguments in gser".to_string());
    }
    
    let gln = ln_gamma(a);
    if gln.is_nan() {
        return Err("Gamma function returned NaN".to_string());
    }
    
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
            return Ok((sum * (-x + a * x.ln()).exp(), gln));
        }
    }
    
    Err("Maximum iterations exceeded in gser".to_string())
}

/// Continued fraction representation for incomplete gamma function
/// Returns (result, ln Γ(a))
fn gcf(a: f32, x: f32) -> Result<(f32, f32), String> {
    if x < 0.0 || a <= 0.0 {
        return Err("Invalid arguments in gcf".to_string());
    }
    
    let gln = ln_gamma(a);
    if gln.is_nan() {
        return Err("Gamma function returned NaN".to_string());
    }
    
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut an: f32;
    
    for i in 1..=MAX_ITER {
        an = -i as f32 * (i as f32 - a);
        b += 2.0;
        d = an * d + b;
        
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        
        c = b + an / c;
        
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < EPS {
            return Ok((h * (-x + a * x.ln()).exp(), gln));
        }
    }
    
    Err("Maximum iterations exceeded in gcf".to_string())
}

/// Thread-safe cached version for repeated calculations
#[derive(Clone)]
pub struct GammaCache {
    cache: Arc<Mutex<std::collections::HashMap<(u32, u32), f32>>>,
}

impl GammaCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get gammp value with caching (for integer parameters)
    pub fn gammp_cached(&self, a: u32, x: f32) -> Result<f32, String> {
        let key = (a, (x * 1000.0) as u32); // Quantize x for caching
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = gammp(a as f32, x)?;
        cache.insert(key, result);
        Ok(result)
    }
    
    /// Get gammq value with caching (for integer parameters)
    pub fn gammq_cached(&self, a: u32, x: f32) -> Result<f32, String> {
        let key = (a, (x * 1000.0) as u32);
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = gammq(a as f32, x)?;
        cache.insert(key, result);
        Ok(result)
    }
}

/// Parallel computation of incomplete gamma functions for multiple inputs
pub fn gammp_batch_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gammp(a, x))
        .collect()
}

pub fn gammq_batch_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gammq(a, x))
        .collect()
}

/// Complementary relationship: P(a, x) + Q(a, x) = 1
pub fn validate_complementarity(a: f32, x: f32) -> Result<bool, String> {
    let p = gammp(a, x)?;
    let q = gammq(a, x)?;
    Ok((p + q - 1.0).abs() < 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gammp_basic() {
        // Test known values
        assert_relative_eq!(gammp(1.0, 1.0).unwrap(), 1.0 - E.recip(), epsilon = 1e-6);
        assert_relative_eq!(gammp(2.0, 1.0).unwrap(), 1.0 - 2.0 * E.recip(), epsilon = 1e-6);
        assert_relative_eq!(gammp(0.5, 0.5).unwrap(), 0.682689, epsilon = 1e-6);
    }

    #[test]
    fn test_gammq_basic() {
        // Test known values
        assert_relative_eq!(gammq(1.0, 1.0).unwrap(), E.recip(), epsilon = 1e-6);
        assert_relative_eq!(gammq(2.0, 1.0).unwrap(), 2.0 * E.recip(), epsilon = 1e-6);
        assert_relative_eq!(gammq(0.5, 0.5).unwrap(), 0.317311, epsilon = 1e-6);
    }

    #[test]
    fn test_complementarity() {
        // P(a, x) + Q(a, x) should equal 1
        let test_cases = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.8), (3.0, 2.0)];
        
        for &(a, x) in &test_cases {
            let p = gammp(a, x).unwrap();
            let q = gammq(a, x).unwrap();
            assert_relative_eq!(p + q, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test edge cases
        assert!(gammp(0.0, 1.0).is_err());
        assert!(gammp(1.0, -1.0).is_err());
        assert!(gammq(0.0, 1.0).is_err());
        assert!(gammq(1.0, -1.0).is_err());
        
        // Test x = 0
        assert_relative_eq!(gammp(1.0, 0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(gammq(1.0, 0.0).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_large_values() {
        // Test with larger values
        assert_relative_eq!(gammp(10.0, 5.0).unwrap(), 0.031828, epsilon = 1e-6);
        assert_relative_eq!(gammq(10.0, 15.0).unwrap(), 0.118597, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_consistency() {
        // Test that parallel batch gives same results as individual calls
        let a_values = [1.0, 2.0, 3.0, 0.5];
        let x_values = [1.0, 1.0, 2.0, 0.5];
        
        let p_results = gammp_batch_parallel(&a_values, &x_values).unwrap();
        let q_results = gammq_batch_parallel(&a_values, &x_values).unwrap();
        
        for i in 0..a_values.len() {
            let p_individual = gammp(a_values[i], x_values[i]).unwrap();
            let q_individual = gammq(a_values[i], x_values[i]).unwrap();
            
            assert_relative_eq!(p_results[i], p_individual, epsilon = 1e-10);
            assert_relative_eq!(q_results[i], q_individual, epsilon = 1e-10);
            assert_relative_eq!(p_results[i] + q_results[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache() {
        let cache = GammaCache::new();
        let result1 = cache.gammp_cached(2, 1.0).unwrap();
        let result2 = cache.gammp_cached(2, 1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, gammp(2.0, 1.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Input slices must have equal length")]
    fn test_batch_length_mismatch() {
        let a = [1.0, 2.0];
        let x = [1.0];
        gammp_batch_parallel(&a, &x).unwrap();
    }

    #[test]
    fn test_large_batch() {
        let n = 500;
        let a_values: Vec<f32> = (1..=n).map(|x| x as f32 / 10.0).collect();
        let x_values: Vec<f32> = (1..=n).map(|x| x as f32 / 5.0).collect();
        
        let results = gammp_batch_parallel(&a_values, &x_values).unwrap();
        
        assert_eq!(results.len(), n);
        // All results should be between 0 and 1
        assert!(results.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_gammp(c: &mut Criterion) {
        c.bench_function("gammp(2.5, 3.5)", |b| {
            b.iter(|| gammp(black_box(2.5), black_box(3.5)).unwrap())
        });
        
        c.bench_function("gammp(10.0, 5.0)", |b| {
            b.iter(|| gammp(black_box(10.0), black_box(5.0)).unwrap())
        });
    }

    fn bench_gammq(c: &mut Criterion) {
        c.bench_function("gammq(2.5, 3.5)", |b| {
            b.iter(|| gammq(black_box(2.5), black_box(3.5)).unwrap())
        });
    }

    fn bench_batch(c: &mut Criterion) {
        let a_values: Vec<f32> = (1..100).map(|x| x as f32 / 2.0).collect();
        let x_values: Vec<f32> = (1..100).map(|x| x as f32 / 3.0).collect();
        
        c.bench_function("gammp_batch_parallel(100)", |b| {
            b.iter(|| gammp_batch_parallel(black_box(&a_values), black_box(&x_values)).unwrap())
        });
    }

    criterion_group!(benches, bench_gammp, bench_gammq, bench_batch);
    criterion_main!(benches);
}
