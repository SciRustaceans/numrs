use std::f64::consts::EULER;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const MAXIT: usize = 100;
const FPMIN: f64 = 1.0e-30;
const EPS: f64 = 6.0e-8;

/// Exponential integral function Ei(x)
/// Computes ∫_{-∞}^x (e^t / t) dt for x > 0
pub fn ei(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err(format!("Bad argument: x must be > 0, got {}", x));
    }

    if x < FPMIN {
        return Ok(x.ln() + EULER);
    }

    if x <= -EPS.ln() {
        ei_series(x)
    } else {
        ei_continued_fraction(x)
    }
}

/// Series representation for small x (x ≤ -ln(EPS))
fn ei_series(x: f64) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut fact = 1.0;

    for k in 1..=MAXIT {
        fact *= x / k as f64;
        let term = fact / k as f64;
        sum += term;

        if term < EPS * sum {
            return Ok(sum + x.ln() + EULER);
        }
    }

    Err(format!("Series failed to converge for x={}", x))
}

/// Continued fraction representation for large x (x > -ln(EPS))
fn ei_continued_fraction(x: f64) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut term = 1.0;

    for k in 1..=MAXIT {
        let prev = term;
        term *= k as f64 / x;

        if term < EPS {
            break;
        }

        if term < prev {
            sum += term;
        } else {
            // Term started increasing, subtract previous term and break
            sum -= prev;
            break;
        }
    }

    Ok(x.exp() * (1.0 + sum) / x)
}

/// Thread-safe cache for Ei values
#[derive(Clone)]
pub struct EiCache {
    cache: Arc<Mutex<std::collections::HashMap<i32, f64>>>,
}

impl EiCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get Ei value with caching (quantized input)
    pub fn get(&self, x: f64) -> Result<f64, String> {
        let key = (x * 1000.0) as i32;
        let mut cache = self.cache.lock().unwrap();

        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }

        let result = ei(x)?;
        cache.insert(key, result);
        Ok(result)
    }

    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.lock().unwrap().is_empty()
    }
}

/// Parallel batch computation for multiple x values
pub fn ei_batch_parallel(x_values: &[f64]) -> Vec<Result<f64, String>> {
    x_values.par_iter()
        .map(|&x| ei(x))
        .collect()
}

/// Compute Ei for a range of values in parallel
pub fn ei_range_parallel(start: f64, end: f64, num_points: usize) -> Vec<(f64, Result<f64, String>)> {
    let step = (end - start) / (num_points - 1) as f64;
    let x_values: Vec<f64> = (0..num_points)
        .map(|i| start + i as f64 * step)
        .collect();

    x_values.par_iter()
        .map(|&x| (x, ei(x)))
        .collect()
}

/// Asymptotic expansion for very large x
pub fn ei_asymptotic(x: f64) -> Result<f64, String> {
    if x <= 10.0 {
        return Err("Asymptotic expansion requires x > 10".to_string());
    }

    let mut sum = 1.0;
    let mut term = 1.0;
    let mut factorial = 1.0;

    for k in 1..=10 {
        factorial *= k as f64;
        term *= k as f64 / x;
        sum += term;

        if term < EPS * sum {
            break;
        }
    }

    Ok(x.exp() * sum / x)
}

/// Smart function that chooses the best algorithm automatically
pub fn ei_smart(x: f64) -> Result<f64, String> {
    if x > 50.0 {
        ei_asymptotic(x).or_else(|_| ei(x))
    } else {
        ei(x)
    }
}

/// Related function: E₁(x) = -Ei(-x) for x > 0
pub fn expint1_via_ei(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be > 0 for E₁(x)".to_string());
    }
    ei(-x).map(|val| -val)
}

/// Compute both Ei(x) and E₁(x) efficiently
pub fn both_exponential_integrals(x: f64) -> Result<(f64, f64), String> {
    let ei_val = ei(x)?;
    let e1_val = if x > 0.0 {
        expint1_via_ei(x)?
    } else {
        return Err("x must be > 0 for E₁(x)".to_string());
    };
    Ok((ei_val, e1_val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ei_basic() {
        // Test known values from mathematical tables
        assert_relative_eq!(ei(1.0).unwrap(), 1.895117816, epsilon = 1e-6);
        assert_relative_eq!(ei(2.0).unwrap(), 4.954234356, epsilon = 1e-6);
        assert_relative_eq!(ei(0.5).unwrap(), 0.454219904, epsilon = 1e-6);
    }

    #[test]
    fn test_ei_small_x() {
        // Test very small x values
        assert_relative_eq!(ei(1e-10).unwrap(), (1e-10).ln() + EULER, epsilon = 1e-6);
        assert_relative_eq!(ei(1e-5).unwrap(), (1e-5).ln() + EULER, epsilon = 1e-6);
    }

    #[test]
    fn test_ei_large_x() {
        // Test large x values
        let result = ei(10.0).unwrap();
        assert!(result > 1000.0 && result.is_finite());
        
        let result = ei(100.0).unwrap();
        assert!(result > 1e40 && result.is_finite());
    }

    #[test]
    fn test_ei_edge_cases() {
        // Test error cases
        assert!(ei(0.0).is_err());
        assert!(ei(-1.0).is_err());
        
        // Test boundary between series and continued fraction
        let boundary = -EPS.ln();
        assert!(ei(boundary - 0.1).is_ok());
        assert!(ei(boundary + 0.1).is_ok());
    }

    #[test]
    fn test_ei_series_convergence() {
        // Test series convergence for small x
        for x in [0.1, 0.5, 1.0, 2.0] {
            if x <= -EPS.ln() {
                assert!(ei_series(x).is_ok());
            }
        }
    }

    #[test]
    fn test_ei_continued_fraction_convergence() {
        // Test continued fraction convergence for large x
        for x in [5.0, 10.0, 20.0, 50.0] {
            if x > -EPS.ln() {
                assert!(ei_continued_fraction(x).is_ok());
            }
        }
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        
        let results = ei_batch_parallel(&x_values);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = ei(x_values[i]).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = EiCache::new();
        
        let result1 = cache.get(1.0).unwrap();
        let result2 = cache.get(1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, ei(1.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_ei_asymptotic() {
        // Test asymptotic expansion for very large x
        let result_asym = ei_asymptotic(20.0).unwrap();
        let result_exact = ei(20.0).unwrap();
        
        assert_relative_eq!(result_asym, result_exact, epsilon = 1e-4);
        
        // Should fail for small x
        assert!(ei_asymptotic(5.0).is_err());
    }

    #[test]
    fn test_expint1_via_ei() {
        // Test relationship: E₁(x) = -Ei(-x)
        let x = 1.0;
        let e1 = expint1_via_ei(x).unwrap();
        let ei_neg = ei(-x).unwrap();
        
        assert_relative_eq!(e1, -ei_neg, epsilon = 1e-10);
        
        // Compare with direct computation
        let e1_direct = super::super::exponential_integral::expint(1, x).unwrap();
        assert_relative_eq!(e1, e1_direct, epsilon = 1e-6);
    }

    #[test]
    fn test_both_exponential_integrals() {
        let (ei_val, e1_val) = both_exponential_integrals(1.0).unwrap();
        
        assert_relative_eq!(ei_val, ei(1.0).unwrap(), epsilon = 1e-10);
        assert_relative_eq!(e1_val, expint1_via_ei(1.0).unwrap(), epsilon = 1e-10);
        
        // Test error case
        assert!(both_exponential_integrals(-1.0).is_err());
    }

    #[test]
    fn test_ei_smart() {
        // Test smart algorithm selection
        let result_small = ei_smart(1.0).unwrap();
        let result_large = ei_smart(100.0).unwrap();
        
        assert_relative_eq!(result_small, ei(1.0).unwrap(), epsilon = 1e-10);
        assert_relative_eq!(result_large, ei(100.0).unwrap(), epsilon = 1e-6);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_ei_small_x(c: &mut Criterion) {
        c.bench_function("ei(0.1)", |b| {
            b.iter(|| ei(black_box(0.1)).unwrap())
        });
        
        c.bench_function("ei(1.0)", |b| {
            b.iter(|| ei(black_box(1.0)).unwrap())
        });
    }

    fn bench_ei_large_x(c: &mut Criterion) {
        c.bench_function("ei(10.0)", |b| {
            b.iter(|| ei(black_box(10.0)).unwrap())
        });
        
        c.bench_function("ei(100.0)", |b| {
            b.iter(|| ei(black_box(100.0)).unwrap())
        });
    }

    fn bench_ei_asymptotic(c: &mut Criterion) {
        c.bench_function("ei_asymptotic(100.0)", |b| {
            b.iter(|| ei_asymptotic(black_box(100.0)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let x_values: Vec<f64> = (1..=100).map(|x| x as f64 / 10.0).collect();
        
        c.bench_function("ei_batch_parallel(100)", |b| {
            b.iter(|| ei_batch_parallel(black_box(&x_values)))
        });
    }

    fn bench_ei_smart(c: &mut Criterion) {
        c.bench_function("ei_smart(1.0)", |b| {
            b.iter(|| ei_smart(black_box(1.0)).unwrap())
        });
        
        c.bench_function("ei_smart(100.0)", |b| {
            b.iter(|| ei_smart(black_box(100.0)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_ei_small_x, 
        bench_ei_large_x,
        bench_ei_asymptotic,
        bench_batch_parallel,
        bench_ei_smart
    );
    criterion_main!(benches);
}
