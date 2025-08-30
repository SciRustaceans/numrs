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

/// Continued fraction representation for the incomplete gamma function Q(a, x)
/// Returns (gammcf, gln) where gammcf = Γ(a, x)/Γ(a) and gln = ln Γ(a)
pub fn gamma_continued_fraction(a: f32, x: f32) -> Result<(f32, f32), String> {
    if x < 0.0 {
        return Err("x must be >= 0 in gamma_continued_fraction".to_string());
    }
    if a <= 0.0 {
        return Err("a must be > 0 in gamma_continued_fraction".to_string());
    }
    
    let gln = ln_gamma(a);
    if gln.is_nan() {
        return Err("Gamma function returned NaN".to_string());
    }
    
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..=MAX_ITER {
        let an = -(i as f32) * (i as f32 - a);
        b += 2.0;
        d = an * d + b;
        
        // Prevent underflow
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        
        c = b + an / c;
        
        // Prevent underflow
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        // Check for convergence
        if (del - 1.0).abs() < EPS {
            let exponent = -x + a * x.ln() - gln;
            let gammcf = exponent.exp() * h;
            return Ok((gammcf, gln));
        }
    }
    
    Err(format!("Maximum iterations ({}) exceeded in gamma_continued_fraction for a={}, x={}", MAX_ITER, a, x))
}

/// Compute the regularized upper incomplete gamma function using continued fraction
pub fn regularized_gamma_continued_fraction(a: f32, x: f32) -> Result<f32, String> {
    gamma_continued_fraction(a, x).map(|(gammcf, _)| gammcf)
}

/// Thread-safe version with result caching
#[derive(Clone)]
pub struct GammaContinuedFractionCache {
    cache: Arc<Mutex<std::collections::HashMap<(u32, u32), (f32, f32)>>>,
}

impl GammaContinuedFractionCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get gamma continued fraction result with caching
    pub fn get(&self, a: u32, x: f32) -> Result<(f32, f32), String> {
        let key = (a, (x * 1000.0) as u32); // Quantize x for caching
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }
        
        let result = gamma_continued_fraction(a as f32, x)?;
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

/// Parallel batch computation of gamma continued fraction for multiple inputs
pub fn gamma_continued_fraction_batch_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<(f32, f32)>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gamma_continued_fraction(a, x))
        .collect()
}

/// Compute only the gammcf values in parallel (discarding gln)
pub fn gamma_continued_fraction_values_parallel(a_values: &[f32], x_values: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a_values.len(), x_values.len(), "Input slices must have equal length");
    
    a_values.par_iter()
        .zip(x_values.par_iter())
        .map(|(&a, &x)| gamma_continued_fraction(a, x).map(|(gammcf, _)| gammcf))
        .collect()
}

/// Optimized version for large x values (where continued fraction excels)
pub fn gamma_continued_fraction_large_x(a: f32, x: f32) -> Result<f32, String> {
    if x < a + 1.0 {
        return Err("x should be >= a + 1 for large_x optimization".to_string());
    }
    
    gamma_continued_fraction(a, x).map(|(gammcf, _)| gammcf)
}

/// Combined function that automatically chooses the best algorithm
pub fn upper_incomplete_gamma(a: f32, x: f32) -> Result<f32, String> {
    if x < a + 1.0 {
        // Use series expansion for x < a + 1
        super::gamma_series::gamma_series(a, x)
            .map(|(gamser, _)| 1.0 - gamser)
    } else {
        // Use continued fraction for x >= a + 1
        gamma_continued_fraction(a, x).map(|(gammcf, _)| gammcf)
    }
}

/// Compute both P(a, x) and Q(a, x) efficiently
pub fn both_incomplete_gammas(a: f32, x: f32) -> Result<(f32, f32), String> {
    if x < a + 1.0 {
        // Compute P via series, then Q = 1 - P
        let (p, gln) = super::gamma_series::gamma_series(a, x)?;
        Ok((p, 1.0 - p))
    } else {
        // Compute Q via continued fraction, then P = 1 - Q
        let (q, gln) = gamma_continued_fraction(a, x)?;
        Ok((1.0 - q, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_continued_fraction_basic() {
        // Test known values
        let (gammcf, gln) = gamma_continued_fraction(1.0, 2.0).unwrap();
        assert_relative_eq!(gammcf, E.powf(-2.0), epsilon = 1e-6);
        
        let (gammcf, _) = gamma_continued_fraction(2.0, 3.0).unwrap();
        assert_relative_eq!(gammcf, (1.0 + 3.0) * E.powf(-3.0), epsilon = 1e-6);
    }

    #[test]
    fn test_gamma_continued_fraction_edge_cases() {
        // Test error cases
        assert!(gamma_continued_fraction(0.0, 1.0).is_err());
        assert!(gamma_continued_fraction(1.0, -1.0).is_err());
        
        // Test very large x (should converge quickly)
        let result = gamma_continued_fraction(2.0, 100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gamma_continued_fraction_convergence() {
        // Test that continued fraction converges for various values
        let test_cases = [
            (2.0, 5.0),   // x > a + 1
            (1.0, 3.0),   // x > a + 1
            (0.5, 2.0),   // x > a + 1
            (10.0, 20.0), // Larger values
        ];
        
        for &(a, x) in &test_cases {
            let result = gamma_continued_fraction(a, x);
            assert!(result.is_ok(), "Failed for a={}, x={}: {:?}", a, x, result.err());
            
            let (gammcf, gln) = result.unwrap();
            assert!(gammcf >= 0.0 && gammcf <= 1.0, "gammcf out of range: {}", gammcf);
            assert!(gln.is_finite(), "gln should be finite: {}", gln);
        }
    }

    #[test]
    fn test_regularized_gamma_continued_fraction() {
        let regularized = regularized_gamma_continued_fraction(2.0, 3.0).unwrap();
        let (full, _) = gamma_continued_fraction(2.0, 3.0).unwrap();
        
        assert_relative_eq!(regularized, full, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_batch() {
        let a_values = [2.0, 3.0, 4.0, 5.0];
        let x_values = [5.0, 6.0, 8.0, 10.0]; // All x > a + 1
        
        let results = gamma_continued_fraction_batch_parallel(&a_values, &x_values).unwrap();
        
        for i in 0..a_values.len() {
            let individual = gamma_continued_fraction(a_values[i], x_values[i]).unwrap();
            assert_relative_eq!(results[i].0, individual.0, epsilon = 1e-10);
            assert_relative_eq!(results[i].1, individual.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = GammaContinuedFractionCache::new();
        
        // First call should compute
        let result1 = cache.get(2, 3.0).unwrap();
        
        // Second call should use cache
        let result2 = cache.get(2, 3.0).unwrap();
        
        assert_relative_eq!(result1.0, result2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.1, result2.1, epsilon = 1e-10);
        
        // Verify cache size
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_upper_incomplete_gamma() {
        // Test automatic algorithm selection
        let q1 = upper_incomplete_gamma(2.0, 1.0).unwrap(); // x < a + 1, should use series
        let q2 = upper_incomplete_gamma(2.0, 5.0).unwrap(); // x > a + 1, should use continued fraction
        
        assert!(q1 > 0.0 && q1 < 1.0);
        assert!(q2 > 0.0 && q2 < 1.0);
        
        // Both should give consistent results with direct calls
        let (p_series, _) = super::super::gamma_series::gamma_series(2.0, 1.0).unwrap();
        let q_series = 1.0 - p_series;
        assert_relative_eq!(q1, q_series, epsilon = 1e-10);
        
        let (q_cf, _) = gamma_continued_fraction(2.0, 5.0).unwrap();
        assert_relative_eq!(q2, q_cf, epsilon = 1e-10);
    }

    #[test]
    fn test_both_incomplete_gammas() {
        let (p, q) = both_incomplete_gammas(2.0, 1.0).unwrap(); // Uses series
        assert_relative_eq!(p + q, 1.0, epsilon = 1e-10);
        
        let (p, q) = both_incomplete_gammas(2.0, 5.0).unwrap(); // Uses continued fraction
        assert_relative_eq!(p + q, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Input slices must have equal length")]
    fn test_batch_length_mismatch() {
        let a = [1.0, 2.0];
        let x = [1.0];
        gamma_continued_fraction_batch_parallel(&a, &x).unwrap();
    }

    #[test]
    fn test_large_batch_performance() {
        let n = 500;
        let a_values: Vec<f32> = (1..=n).map(|x| x as f32 / 2.0).collect();
        let x_values: Vec<f32> = (1..=n).map(|x| (x + 5) as f32).collect(); // Ensure x > a + 1
        
        let results = gamma_continued_fraction_values_parallel(&a_values, &x_values).unwrap();
        
        assert_eq!(results.len(), n);
        assert!(results.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_gamma_continued_fraction(c: &mut Criterion) {
        c.bench_function("gamma_continued_fraction(2.5, 5.0)", |b| {
            b.iter(|| gamma_continued_fraction(black_box(2.5), black_box(5.0)).unwrap())
        });
        
        c.bench_function("gamma_continued_fraction(10.0, 20.0)", |b| {
            b.iter(|| gamma_continued_fraction(black_box(10.0), black_box(20.0)).unwrap())
        });
    }

    fn bench_upper_incomplete_gamma(c: &mut Criterion) {
        c.bench_function("upper_incomplete_gamma(2.5, 1.5)", |b| {
            b.iter(|| upper_incomplete_gamma(black_box(2.5), black_box(1.5)).unwrap())
        });
        
        c.bench_function("upper_incomplete_gamma(2.5, 5.0)", |b| {
            b.iter(|| upper_incomplete_gamma(black_box(2.5), black_box(5.0)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let a_values: Vec<f32> = (1..100).map(|x| x as f32 / 2.0).collect();
        let x_values: Vec<f32> = (1..100).map(|x| (x + 10) as f32).collect(); // x > a + 1
        
        c.bench_function("gamma_continued_fraction_batch_parallel(100)", |b| {
            b.iter(|| gamma_continued_fraction_batch_parallel(black_box(&a_values), black_box(&x_values)).unwrap())
        });
    }

    fn bench_both_incomplete_gammas(c: &mut Criterion) {
        c.bench_function("both_incomplete_gammas(2.5, 1.5)", |b| {
            b.iter(|| both_incomplete_gammas(black_box(2.5), black_box(1.5)).unwrap())
        });
        
        c.bench_function("both_incomplete_gammas(2.5, 5.0)", |b| {
            b.iter(|| both_incomplete_gammas(black_box(2.5), black_box(5.0)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_gamma_continued_fraction, 
        bench_upper_incomplete_gamma,
        bench_batch_parallel,
        bench_both_incomplete_gammas
    );
    criterion_main!(benches);
}
