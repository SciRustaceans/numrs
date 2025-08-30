use std::f32::consts::{E, LN_2, PI};
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

/// Series representation for the incomplete gamma function P(a, x)
/// Returns (gamser, gln) where gamser = γ(a, x)/Γ(a) and gln = ln Γ(a)
pub fn gamma_series(a: f32, x: f32) -> Result<(f32, f32), String> {
    if x < 0.0 {
        return Err("x must be >= 0 in gamma_series".to_string());
    }
    if a <= 0.0 {
        return Err("a must be > 0 in gamma_series".to_string());
    }
    
    let gln = ln_gamma(a);
    if gln.is_nan() {
        return Err("Gamma function returned NaN".to_string());
    }
    
    // Handle x = 0 case
    if x == 0.0 {
        return Ok((0.0, gln));
    }
    
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    
    for n in 1..=MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        
        // Check for convergence
        if del.abs() < sum.abs() * EPS {
            let exponent = -x + a * x.ln() - gln;
            let gamser = sum * exponent.exp();
            return Ok((gamser, gln));
        }
    }
    
    Err(format!("Maximum iterations ({}) exceeded in gamma_series for a={}, x={}", MAX_ITER, a, x))
}

/// Thread-safe version with result caching
#[derive(Clone)]
pub struct GammaSeriesCache {
    cache: Arc<Mutex<std::collections::HashMap<(u32, u32), (f32, f32)>>>,
}

impl GammaSeriesCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get gamma series result with caching
    pub fn get(&self, a: u32, x: f32) -> Result<(f32, f32), String> {
        let key = (a, (x * 1000.0) as u32); // Quantize x for caching
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = gamma_series(a as f32, x)?;
        cache.insert(key, result);
        Ok(result)
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        let cache = self.cache.lock().unwrap();
        cache.is_empty()
    }
}

/// Parallel batch computation of gamma series for multiple inputs
pub fn gamma_series_batch_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<(f32, f32)>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gamma_series(a, x))
        .collect()
}

/// Compute only the gamser values in parallel (discarding gln)
pub fn gamma_series_values_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gamma_series(a, x).map(|(gamser, _)| gamser))
        .collect()
}

/// Compute only the ln_gamma values in parallel (discarding gamser)
pub fn ln_gamma_values_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gamma_series(a, x).map(|(_, gln)| gln))
        .collect()
}

/// Optimized version for integer 'a' values
pub fn gamma_series_int_a(a: u32, x: f32) -> Result<f32, String> {
    if x < 0.0 {
        return Err("x must be >= 0".to_string());
    }
    
    if x == 0.0 {
        return Ok(0.0);
    }
    
    // For integer a, we can use a more efficient computation
    let a_f32 = a as f32;
    let mut sum = 1.0 / a_f32;
    let mut del = sum;
    let mut ap = a_f32;
    
    for n in 1..=MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        
        if del.abs() < sum.abs() * EPS {
            // For integer a, Γ(a) = (a-1)!
            let factorial: f32 = (1..a).map(|i| i as f32).product();
            let exponent = -x + a_f32 * x.ln();
            let gamser = sum * exponent.exp() / factorial;
            return Ok(gamser);
        }
    }
    
    Err(format!("Maximum iterations exceeded for integer a={}", a))
}

/// Compute the regularized lower incomplete gamma function using series
pub fn regularized_gamma_series(a: f32, x: f32) -> Result<f32, String> {
    gamma_series(a, x).map(|(gamser, _)| gamser)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_series_basic() {
        // Test known values
        let (gamser, gln) = gamma_series(1.0, 1.0).unwrap();
        assert_relative_eq!(gamser, 1.0 - E.recip(), epsilon = 1e-6);
        assert_relative_eq!(gln, 0.0, epsilon = 1e-6);
        
        let (gamser, _) = gamma_series(2.0, 1.0).unwrap();
        assert_relative_eq!(gamser, 1.0 - 2.0 * E.recip(), epsilon = 1e-6);
    }

    #[test]
    fn test_gamma_series_edge_cases() {
        // Test x = 0
        let (gamser, gln) = gamma_series(1.0, 0.0).unwrap();
        assert_relative_eq!(gamser, 0.0, epsilon = 1e-10);
        assert_relative_eq!(gln, 0.0, epsilon = 1e-10);
        
        // Test error cases
        assert!(gamma_series(0.0, 1.0).is_err());
        assert!(gamma_series(1.0, -1.0).is_err());
    }

    #[test]
    fn test_gamma_series_convergence() {
        // Test that series converges for various values
        let test_cases = [
            (0.5, 0.5),
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 1.0),
        ];
        
        for &(a, x) in &test_cases {
            let result = gamma_series(a, x);
            assert!(result.is_ok(), "Failed for a={}, x={}: {:?}", a, x, result.err());
            
            let (gamser, gln) = result.unwrap();
            assert!(gamser >= 0.0 && gamser <= 1.0, "gamser out of range: {}", gamser);
            assert!(gln.is_finite(), "gln should be finite: {}", gln);
        }
    }

    #[test]
    fn test_gamma_series_int_a() {
        // Test integer a optimization
        let result1 = gamma_series(2, 1.0).unwrap().0;
        let result2 = gamma_series_int_a(2, 1.0).unwrap();
        
        assert_relative_eq!(result1, result2, epsilon = 1e-6);
        
        // Test multiple integer values
        for a in 1..=5 {
            let x = a as f32 / 2.0;
            let result1 = gamma_series(a as f32, x).unwrap().0;
            let result2 = gamma_series_int_a(a, x).unwrap();
            
            assert_relative_eq!(result1, result2, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_parallel_batch() {
        let a_values = [1.0, 2.0, 3.0, 0.5];
        let x_values = [1.0, 1.0, 2.0, 0.5];
        
        let results = gamma_series_batch_parallel(&a_values, &x_values).unwrap();
        
        for i in 0..a_values.len() {
            let individual = gamma_series(a_values[i], x_values[i]).unwrap();
            assert_relative_eq!(results[i].0, individual.0, epsilon = 1e-10);
            assert_relative_eq!(results[i].1, individual.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = GammaSeriesCache::new();
        
        // First call should compute
        let result1 = cache.get(2, 1.0).unwrap();
        
        // Second call should use cache
        let result2 = cache.get(2, 1.0).unwrap();
        
        assert_relative_eq!(result1.0, result2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.1, result2.1, epsilon = 1e-10);
        
        // Verify cache size
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        
        // Test clearing cache
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_regularized_gamma_series() {
        let regularized = regularized_gamma_series(1.0, 1.0).unwrap();
        let (full, _) = gamma_series(1.0, 1.0).unwrap();
        
        assert_relative_eq!(regularized, full, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Input slices must have equal length")]
    fn test_batch_length_mismatch() {
        let a = [1.0, 2.0];
        let x = [1.0];
        gamma_series_batch_parallel(&a, &x).unwrap();
    }

    #[test]
    fn test_large_batch_performance() {
        let n = 1000;
        let a_values: Vec<f32> = (1..=n).map(|x| x as f32 / 10.0).collect();
        let x_values: Vec<f32> = (1..=n).map(|x| x as f32 / 5.0).collect();
        
        let results = gamma_series_values_parallel(&a_values, &x_values).unwrap();
        
        assert_eq!(results.len(), n);
        assert!(results.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_gamma_series(c: &mut Criterion) {
        c.bench_function("gamma_series(2.5, 1.5)", |b| {
            b.iter(|| gamma_series(black_box(2.5), black_box(1.5)).unwrap())
        });
        
        c.bench_function("gamma_series(10.0, 5.0)", |b| {
            b.iter(|| gamma_series(black_box(10.0), black_box(5.0)).unwrap())
        });
    }

    fn bench_gamma_series_int(c: &mut Criterion) {
        c.bench_function("gamma_series_int_a(5, 2.5)", |b| {
            b.iter(|| gamma_series_int_a(black_box(5), black_box(2.5)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let a_values: Vec<f32> = (1..100).map(|x| x as f32 / 2.0).collect();
        let x_values: Vec<f32> = (1..100).map(|x| x as f32 / 3.0).collect();
        
        c.bench_function("gamma_series_batch_parallel(100)", |b| {
            b.iter(|| gamma_series_batch_parallel(black_box(&a_values), black_box(&x_values)).unwrap())
        });
    }

    fn bench_ln_gamma_parallel(c: &mut Criterion) {
        let a_values: Vec<f32> = (1..100).map(|x| x as f32 / 2.0).collect();
        let x_values: Vec<f32> = (1..100).map(|x| x as f32 / 3.0).collect();
        
        c.bench_function("ln_gamma_values_parallel(100)", |b| {
            b.iter(|| ln_gamma_values_parallel(black_box(&a_values), black_box(&x_values)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_gamma_series, 
        bench_gamma_series_int, 
        bench_batch_parallel,
        bench_ln_gamma_parallel
    );
    criterion_main!(benches);
}
