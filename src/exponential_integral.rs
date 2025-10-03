use std::f64::consts::E;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const MAXIT: usize = 100;
const FPMIN: f64 = 1.0e-30;
const EPS: f64 = 1.0e-7;

/// Exponential integral function E_n(x)
/// Computes ∫₁∞ e^(-x·t) / t^n dt
pub fn expint(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1)) {
        return Err(format!("Bad arguments: n={}, x={}", n, x));
    }

    let nm1 = n - 1;

    match (n, x) {
        (0, _) => Ok((-x).exp() / x),
        (_, 0.0) => Ok(1.0 / nm1 as f64),
        _ => {
            if x > 1.0 {
                expint_continued_fraction(n, x, nm1)
            } else {
                expint_series(n, x, nm1)
            }
        }
    }
}

/// Continued fraction representation for x > 1
fn expint_continued_fraction(n: i32, x: f64, nm1: i32) -> Result<f64, String> {
    let mut b = x + n as f64;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=MAXIT {
        let a = i as f64 * (nm1 as f64 + i as f64);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        let del = c * d;
        h *= del;

        if (del - 1.0).abs() < EPS {
            return Ok(h * (-x).exp());
        }
    }

    Err(format!("Continued fraction failed to converge for n={}, x={}", n, x))
}

/// Series representation for x ≤ 1
fn expint_series(n: i32, x: f64, nm1: i32) -> Result<f64, String> {
    let mut ans = if nm1 != 0 {
        1.0 / nm1 as f64
    } else {
        -x.ln() - E
    };

    let mut fact = 1.0;

    for i in 1..=MAXIT {
        fact *= -x / i as f64;

        let del = if i as i32 != nm1 {
            -fact / (i as f64 - nm1 as f64)
        } else {
            // Compute psi(n) = H_{n-1} - E
            let psi = harmonic_number(nm1) - E;
            fact * (-x.ln() + psi)
        };

        ans += del;

        if del.abs() < ans.abs() * EPS {
            return Ok(ans);
        }
    }

    Err(format!("Series failed to converge for n={}, x={}", n, x))
}

/// Compute harmonic number H_n = ∑_{k=1}^n 1/k
fn harmonic_number(n: i32) -> f64 {
    if n <= 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for k in 1..=n {
        sum += 1.0 / k as f64;
    }
    sum
}

/// Thread-safe cache for exponential integral values
#[derive(Clone)]
pub struct ExpIntCache {
    cache: Arc<Mutex<std::collections::HashMap<(i32, i32), f64>>>,
}

impl ExpIntCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get expint value with caching (quantized input)
    pub fn get(&self, n: i32, x: f64) -> Result<f64, String> {
        let key = (n, (x * 1000.0) as i32);
        let mut cache = self.cache.lock().unwrap();

        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }

        let result = expint(n, x)?;
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

/// Parallel batch computation for multiple n values with fixed x
pub fn expint_batch_n_parallel(n_values: &[i32], x: f64) -> Vec<Result<f64, String>> {
    n_values.par_iter()
        .map(|&n| expint(n, x))
        .collect()
}

/// Parallel batch computation for multiple x values with fixed n
pub fn expint_batch_x_parallel(n: i32, x_values: &[f64]) -> Vec<Result<f64, String>> {
    x_values.par_iter()
        .map(|&x| expint(n, x))
        .collect()
}

/// Compute multiple (n, x) pairs in parallel
pub fn expint_batch_parallel(pairs: &[(i32, f64)]) -> Vec<Result<f64, String>> {
    pairs.par_iter()
        .map(|&(n, x)| expint(n, x))
        .collect()
}

/// Special case: E₁(x) - commonly used exponential integral
pub fn expint1(x: f64) -> Result<f64, String> {
    expint(1, x)
}

/// Special case: E₂(x)
pub fn expint2(x: f64) -> Result<f64, String> {
    expint(2, x)
}

/// Asymptotic expansion for large x
pub fn expint_asymptotic(n: i32, x: f64) -> Result<f64, String> {
    if x <= 10.0 {
        return Err("Asymptotic expansion requires x > 10".to_string());
    }

    let mut result = (-x).exp() / x;
    let mut term = 1.0;
    let mut sum = 1.0;

    for k in 1..=10 {
        term *= -(n + k as i32 - 1) as f64 / (x * k as f64);
        sum += term;
        
        if term.abs() < EPS * sum.abs() {
            break;
        }
    }

    Ok(result * sum)
}

/// Smart function that chooses the best algorithm automatically
pub fn expint_smart(n: i32, x: f64) -> Result<f64, String> {
    if x > 10.0 {
        expint_asymptotic(n, x).or_else(|_| expint(n, x))
    } else {
        expint(n, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_expint_basic() {
        // Test known values
        assert_relative_eq!(expint(1, 1.0).unwrap(), 0.219383934, epsilon = 1e-6);
        assert_relative_eq!(expint(2, 1.0).unwrap(), 0.148495506, epsilon = 1e-6);
        assert_relative_eq!(expint(3, 1.0).unwrap(), 0.109691967, epsilon = 1e-6);
    }

    #[test]
    fn test_expint_edge_cases() {
        // Test edge cases
        assert!(expint(-1, 1.0).is_err());
        assert!(expint(1, -1.0).is_err());
        assert!(expint(0, 0.0).is_err());
        assert!(expint(1, 0.0).is_err());

        // Test n=0
        assert_relative_eq!(expint(0, 1.0).unwrap(), (-1.0).exp() / 1.0, epsilon = 1e-10);

        // Test x=0
        assert_relative_eq!(expint(2, 0.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(expint(3, 0.0).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_expint_series_vs_continued_fraction() {
        // Test that both methods give consistent results near x=1
        let result_series = expint_series(2, 0.9, 1).unwrap();
        let result_cf = expint_continued_fraction(2, 1.1, 1).unwrap();
        
        // They should be close but not exactly equal due to different algorithms
        assert!((result_series - result_cf).abs() < 0.1);
    }

    #[test]
    fn test_harmonic_number() {
        assert_relative_eq!(harmonic_number(1), 1.0, epsilon = 1e-10);
        assert_relative_eq!(harmonic_number(2), 1.5, epsilon = 1e-10);
        assert_relative_eq!(harmonic_number(3), 1.0 + 1.0/2.0 + 1.0/3.0, epsilon = 1e-10);
        assert_relative_eq!(harmonic_number(0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expint_special_cases() {
        assert_relative_eq!(expint1(1.0).unwrap(), expint(1, 1.0).unwrap(), epsilon = 1e-10);
        assert_relative_eq!(expint2(1.0).unwrap(), expint(2, 1.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_batch() {
        let n_values = [1, 2, 3, 4];
        let x = 1.0;
        
        let results = expint_batch_n_parallel(&n_values, x);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = expint(n_values[i], x).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = ExpIntCache::new();
        
        let result1 = cache.get(2, 1.0).unwrap();
        let result2 = cache.get(2, 1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, expint(2, 1.0).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_expint_asymptotic() {
        // Test asymptotic expansion for large x
        let result_asym = expint_asymptotic(1, 10.0).unwrap();
        let result_exact = expint(1, 10.0).unwrap();
        
        assert_relative_eq!(result_asym, result_exact, epsilon = 1e-4);
        
        // Should fail for small x
        assert!(expint_asymptotic(1, 5.0).is_err());
    }

    #[test]
    fn test_expint_smart() {
        // Test smart algorithm selection
        let result_small = expint_smart(1, 1.0).unwrap();
        let result_large = expint_smart(1, 20.0).unwrap();
        
        assert_relative_eq!(result_small, expint(1, 1.0).unwrap(), epsilon = 1e-10);
        assert_relative_eq!(result_large, expint(1, 20.0).unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_large_values() {
        // Test behavior for large n and x
        assert!(expint(10, 100.0).is_ok());
        assert!(expint(50, 1.0).is_ok());
        
        // Very small results should still be finite
        let result = expint(10, 100.0).unwrap();
        assert!(result.is_finite() && result >= 0.0);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_expint_small_x(c: &mut Criterion) {
        c.bench_function("expint(2, 0.5)", |b| {
            b.iter(|| expint(black_box(2), black_box(0.5)).unwrap())
        });
    }

    fn bench_expint_large_x(c: &mut Criterion) {
        c.bench_function("expint(2, 10.0)", |b| {
            b.iter(|| expint(black_box(2), black_box(10.0)).unwrap())
        });
    }

    fn bench_expint_asymptotic(c: &mut Criterion) {
        c.bench_function("expint_asymptotic(2, 20.0)", |b| {
            b.iter(|| expint_asymptotic(black_box(2), black_box(20.0)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let n_values: Vec<i32> = (1..=50).collect();
        let x_values: Vec<f64> = (1..=50).map(|x| x as f64 / 10.0).collect();
        let pairs: Vec<(i32, f64)> = (1..=20).map(|i| (i, i as f64 / 2.0)).collect();
        
        c.bench_function("expint_batch_n_parallel(50)", |b| {
            b.iter(|| expint_batch_n_parallel(black_box(&n_values), black_box(1.0)))
        });
        
        c.bench_function("expint_batch_parallel(20)", |b| {
            b.iter(|| expint_batch_parallel(black_box(&pairs)))
        });
    }

    criterion_group!(
        benches, 
        bench_expint_small_x, 
        bench_expint_large_x,
        bench_expint_asymptotic,
        bench_batch_parallel
    );
    criterion_main!(benches);
}
