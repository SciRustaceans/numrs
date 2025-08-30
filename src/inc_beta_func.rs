use std::f64::consts::{E, PI};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const MAXIT: usize = 100;
const FPMIN: f64 = 1.0e-30;
const EPS: f64 = 3.0e-7;

/// Calculates the natural logarithm of the gamma function using Lanczos approximation
pub fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation coefficients for f64 precision
    const COEFFS: [f64; 7] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
    ];
    
    if x <= 0.0 {
        return f64::NAN;
    }
    
    // Reflection formula for x < 0.5
    if x < 0.5 {
        let pi = PI;
        return (pi / ((pi * x).sin())).ln() - ln_gamma(1.0 - x);
    }
    
    let mut x = x - 1.0;
    let mut t = COEFFS[0];
    
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        t += coeff / (x + i as f64);
    }
    
    let sqrt_2pi = (2.0 * PI).sqrt();
    (sqrt_2pi * t).ln() + (x + 0.5) * (x + 1.5 + 5.5).ln() - (x + 1.5 + 5.5)
}

/// Continued fraction evaluation for incomplete beta function
fn betacf(a: f64, b: f64, x: f64) -> Result<f64, String> {
    if x < 0.0 || x > 1.0 {
        return Err("x must be in [0, 1] in betacf".to_string());
    }
    if a <= 0.0 || b <= 0.0 {
        return Err("a and b must be > 0 in betacf".to_string());
    }

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    
    d = 1.0 / d;
    let mut h = d;
    
    for i in 1..=MAXIT {
        let m2 = 2 * i;
        let i_f64 = i as f64;
        
        // Even step
        let mut aa = i_f64 * (b - i_f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        
        d = 1.0 / d;
        h *= d * c;
        
        // Odd step
        aa = -(a + i_f64) * (qab + i_f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < EPS {
            return Ok(h);
        }
    }
    
    Err(format!("Continued fraction failed to converge in betacf for a={}, b={}, x={}", a, b, x))
}

/// Regularized incomplete beta function I_x(a, b)
/// Computes the cumulative distribution function for the beta distribution
pub fn betai(a: f64, b: f64, x: f64) -> Result<f64, String> {
    if x < 0.0 || x > 1.0 {
        return Err(format!("x must be in [0, 1], got {}", x));
    }
    if a <= 0.0 || b <= 0.0 {
        return Err(format!("a and b must be > 0, got a={}, b={}", a, b));
    }

    // Handle edge cases
    if x == 0.0 || x == 1.0 {
        return Ok(x);
    }

    // Compute the prefactor bt
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // Choose the most efficient computation path
    if x < (a + 1.0) / (a + b + 2.0) {
        let cf = betacf(a, b, x)?;
        Ok(bt * cf / a)
    } else {
        let cf = betacf(b, a, 1.0 - x)?;
        Ok(1.0 - bt * cf / b)
    }
}

/// Thread-safe cache for incomplete beta values
#[derive(Clone)]
pub struct BetaICache {
    cache: Arc<Mutex<std::collections::HashMap<(i32, i32, i32), f64>>>,
}

impl BetaICache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get betai value with caching (quantized inputs)
    pub fn get(&self, a: f64, b: f64, x: f64) -> Result<f64, String> {
        let key = (
            (a * 1000.0) as i32,
            (b * 1000.0) as i32,
            (x * 1000.0) as i32,
        );
        let mut cache = self.cache.lock().unwrap();

        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }

        let result = betai(a, b, x)?;
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

/// Parallel batch computation for multiple parameter sets
pub fn betai_batch_parallel(params: &[(f64, f64, f64)]) -> Vec<Result<f64, String>> {
    params.par_iter()
        .map(|&(a, b, x)| betai(a, b, x))
        .collect()
}

/// Compute incomplete beta for a grid of values
pub fn betai_grid_parallel(a_values: &[f64], b_values: &[f64], x: f64) -> Vec<Vec<Result<f64, String>>> {
    a_values.par_iter()
        .map(|&a| {
            b_values.iter()
                .map(|&b| betai(a, b, x))
                .collect()
        })
        .collect()
}

/// Special case: Regularized incomplete beta for integer parameters
pub fn betai_int(a: i32, b: i32, x: f64) -> Result<f64, String> {
    if a <= 0 || b <= 0 {
        return Err("a and b must be positive integers".to_string());
    }
    betai(a as f64, b as f64, x)
}

/// Inverse incomplete beta function (quantile function)
pub fn betai_inv(a: f64, b: f64, p: f64) -> Result<f64, String> {
    if p < 0.0 || p > 1.0 {
        return Err("p must be in [0, 1]".to_string());
    }

    // Use Newton's method
    let mut x = if p < 0.5 {
        (p * a as f64).powf(1.0 / a)
    } else {
        1.0 - ((1.0 - p) * b as f64).powf(1.0 / b)
    };

    for _ in 0..20 {
        let f = betai(a, b, x)? - p;
        if f.abs() < 1e-10 {
            break;
        }

        // Derivative: x^(a-1) * (1-x)^(b-1) / B(a, b)
        let df = x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0) / (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)).exp();
        
        if df.abs() < 1e-10 {
            break;
        }

        x -= f / df;
        x = x.clamp(0.0, 1.0);
    }

    Ok(x)
}

/// Beta distribution cumulative distribution function
pub fn beta_cdf(a: f64, b: f64, x: f64) -> Result<f64, String> {
    betai(a, b, x)
}

/// Beta distribution probability density function
pub fn beta_pdf(a: f64, b: f64, x: f64) -> Result<f64, String> {
    if x < 0.0 || x > 1.0 {
        return Ok(0.0);
    }
    
    let norm = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)).exp();
    Ok(x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0) / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_betai_basic() {
        // Test known values
        assert_relative_eq!(betai(1.0, 1.0, 0.5).unwrap(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(betai(2.0, 2.0, 0.5).unwrap(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(betai(0.5, 0.5, 0.5).unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_betai_edge_cases() {
        // Test edge cases
        assert_relative_eq!(betai(2.0, 3.0, 0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(betai(2.0, 3.0, 1.0).unwrap(), 1.0, epsilon = 1e-10);
        
        // Test error cases
        assert!(betai(2.0, 3.0, -0.1).is_err());
        assert!(betai(2.0, 3.0, 1.1).is_err());
        assert!(betai(0.0, 3.0, 0.5).is_err());
        assert!(betai(2.0, 0.0, 0.5).is_err());
    }

    #[test]
    fn test_betai_symmetry() {
        // Test symmetry: I_x(a, b) = 1 - I_{1-x}(b, a)
        let a = 2.0;
        let b = 3.0;
        let x = 0.3;
        
        let result1 = betai(a, b, x).unwrap();
        let result2 = 1.0 - betai(b, a, 1.0 - x).unwrap();
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
    }

    #[test]
    fn test_betai_special_cases() {
        // Test special cases
        assert_relative_eq!(betai(1.0, 1.0, 0.7).unwrap(), 0.7, epsilon = 1e-10);
        assert_relative_eq!(betai(1.0, 2.0, 0.5).unwrap(), 0.75, epsilon = 1e-6);
        assert_relative_eq!(betai(2.0, 1.0, 0.5).unwrap(), 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_betacf_convergence() {
        // Test continued fraction convergence
        assert!(betacf(2.0, 3.0, 0.3).is_ok());
        assert!(betacf(0.5, 0.5, 0.5).is_ok());
        assert!(betacf(10.0, 10.0, 0.7).is_ok());
    }

    #[test]
    fn test_parallel_batch() {
        let params = [
            (1.0, 1.0, 0.5),
            (2.0, 2.0, 0.5),
            (0.5, 0.5, 0.5),
            (2.0, 3.0, 0.3),
        ];
        
        let results = betai_batch_parallel(&params);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let (a, b, x) = params[i];
            let expected = betai(a, b, x).unwrap();
            assert_relative_eq!(result.as_ref().unwrap(), &expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = BetaICache::new();
        
        let result1 = cache.get(2.0, 3.0, 0.5).unwrap();
        let result2 = cache.get(2.0, 3.0, 0.5).unwrap(); // Should be cached
        
        assert_relative_eq!(result1, result2, epsilon = 1e-10);
        assert_relative_eq!(result1, betai(2.0, 3.0, 0.5).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_betai_int() {
        assert_relative_eq!(betai_int(2, 3, 0.5).unwrap(), betai(2.0, 3.0, 0.5).unwrap(), epsilon = 1e-10);
        assert!(betai_int(0, 3, 0.5).is_err());
    }

    #[test]
    fn test_betai_inv() {
        // Test inverse function
        let a = 2.0;
        let b = 3.0;
        let x = 0.3;
        
        let p = betai(a, b, x).unwrap();
        let x_inv = betai_inv(a, b, p).unwrap();
        
        assert_relative_eq!(x, x_inv, epsilon = 1e-6);
    }

    #[test]
    fn test_beta_distribution() {
        // Test PDF and CDF relationship
        let a = 2.0;
        let b = 3.0;
        let x = 0.4;
        
        let cdf = beta_cdf(a, b, x).unwrap();
        let pdf = beta_pdf(a, b, x).unwrap();
        
        assert!(cdf >= 0.0 && cdf <= 1.0);
        assert!(pdf >= 0.0);
        
        // PDF should integrate to CDF (roughly)
        let dx = 0.01;
        let mut integral = 0.0;
        for i in 0..100 {
            let x_i = i as f64 * dx;
            integral += beta_pdf(a, b, x_i).unwrap() * dx;
        }
        
        assert_relative_eq!(integral, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_large_parameters() {
        // Test with larger parameters
        assert!(betai(10.0, 10.0, 0.5).is_ok());
        assert!(betai(100.0, 100.0, 0.5).is_ok());
        
        // Results should be symmetric and finite
        let result = betai(50.0, 50.0, 0.5).unwrap();
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_betai_small_params(c: &mut Criterion) {
        c.bench_function("betai(2.0, 3.0, 0.3)", |b| {
            b.iter(|| betai(black_box(2.0), black_box(3.0), black_box(0.3)).unwrap())
        });
    }

    fn bench_betai_large_params(c: &mut Criterion) {
        c.bench_function("betai(10.0, 10.0, 0.5)", |b| {
            b.iter(|| betai(black_box(10.0), black_box(10.0), black_box(0.5)).unwrap())
        });
    }

    fn bench_betai_extreme(c: &mut Criterion) {
        c.bench_function("betai(0.5, 0.5, 0.5)", |b| {
            b.iter(|| betai(black_box(0.5), black_box(0.5), black_box(0.5)).unwrap())
        });
    }

    fn bench_batch_parallel(c: &mut Criterion) {
        let params: Vec<(f64, f64, f64)> = (1..=20)
            .map(|i| (i as f64, (i + 1) as f64, 0.5))
            .collect();
        
        c.bench_function("betai_batch_parallel(20)", |b| {
            b.iter(|| betai_batch_parallel(black_box(&params)))
        });
    }

    fn bench_betai_inv(c: &mut Criterion) {
        c.bench_function("betai_inv(2.0, 3.0, 0.3)", |b| {
            b.iter(|| betai_inv(black_box(2.0), black_box(3.0), black_box(0.3)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_betai_small_params, 
        bench_betai_large_params,
        bench_betai_extreme,
        bench_batch_parallel,
        bench_betai_inv
    );
    criterion_main!(benches);
}
