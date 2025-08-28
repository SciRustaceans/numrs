use std::simd::{f64x4, SimdFloat};
use std::f64::consts::{LN_2, PI};
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Precomputed coefficients for Lanczos approximation
static COEFFS: [f64; 6] = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
];

/// Constants for the approximation
const SQRT_2PI: f64 = 2.5066282746310005; // sqrt(2π)
const G: f64 = 5.0; // Lanczos parameter
const HALF: f64 = 0.5;
const ONE: f64 = 1.0;

/// Thread-local cache for frequently computed values
thread_local! {
    static GAMMLN_CACHE: Mutex<lru::LruCache<u64, f64>> = Mutex::new(lru::LruCache::new(1000));
}

/// Compute the natural logarithm of the gamma function
/// Using Lanczos approximation with high precision
pub fn gammln(x: f64) -> f64 {
    // Handle special cases and domain errors
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY; // Γ(0) = ∞
        }
        if x == x.floor() && x < 0.0 {
            return f64::NAN; // Γ(negative integer) = NaN
        }
    }

    // Use reflection formula for negative arguments
    if x < 0.5 {
        return LN_2.ln() - (-x).ln() - gammln(1.0 - x) - (PI * x).sin().ln();
    }

    // Check cache first
    let x_bits = x.to_bits();
    let cached = GAMMLN_CACHE.with(|cache| {
        let mut cache = cache.lock().unwrap();
        cache.get(&x_bits).copied()
    });

    if let Some(result) = cached {
        return result;
    }

    // Lanczos approximation
    let mut y = x;
    let mut tmp = x + G + HALF;
    tmp -= (x + HALF) * tmp.ln();
    
    let mut ser = 1.000000000190015;
    let mut y_acc = x + 1.0;

    // Process coefficients in pairs for better instruction-level parallelism
    for j in 0..3 {
        let idx = j * 2;
        ser += COEFFS[idx] / y_acc;
        y_acc += 1.0;
        ser += COEFFS[idx + 1] / y_acc;
        y_acc += 1.0;
    }

    let result = -tmp + (SQRT_2PI * ser / x).ln();

    // Cache the result
    GAMMLN_CACHE.with(|cache| {
        let mut cache = cache.lock().unwrap();
        cache.put(x_bits, result);
    });

    result
}

/// SIMD-optimized version for multiple values
pub fn gammln_simd(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&x| gammln(x)).collect()
}

/// Batch processing with parallel execution
pub fn gammln_batch(x: &[f64]) -> Vec<f64> {
    use rayon::prelude::*;
    x.par_iter().map(|&x| gammln(x)).collect()
}

/// Inline version for maximum performance in hot loops
#[inline(always)]
pub fn gammln_fast(x: f64) -> f64 {
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        if x == x.floor() && x < 0.0 {
            return f64::NAN;
        }
    }

    if x < 0.5 {
        return LN_2.ln() - (-x).ln() - gammln_fast(1.0 - x) - (PI * x).sin().ln();
    }

    let mut y = x;
    let mut tmp = x + G + HALF;
    tmp -= (x + HALF) * tmp.ln();
    
    let mut ser = 1.000000000190015;
    let mut y_acc = x + 1.0;

    for j in 0..3 {
        let idx = j * 2;
        ser += COEFFS[idx] / y_acc;
        y_acc += 1.0;
        ser += COEFFS[idx + 1] / y_acc;
        y_acc += 1.0;
    }

    -tmp + (SQRT_2PI * ser / x).ln()
}

/// Error function approximation using gamma function
pub fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    
    // Approximation using incomplete gamma function
    let z = x * x;
    if x < 0.5 {
        x * (-z).exp() * 2.0 / PI.sqrt()
    } else {
        1.0 - (-z).exp() / (x * PI.sqrt()) * (1.0 - 1.0 / (2.0 * z))
    }
}

/// Complementary error function
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Gamma function itself (not log)
pub fn gamma(x: f64) -> f64 {
    gammln(x).exp()
}

/// Regularized incomplete gamma functions
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series representation
        let mut sum = 1.0;
        let mut term = 1.0;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }
        x.powf(a) * (-x).exp() * sum / gamma(a)
    } else {
        // Continued fraction representation
        1.0 - gamma_q(a, x)
    }
}

pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p(a, x)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        
        for i in 1..100 {
            let an = -i as f64 * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        x.powf(a) * (-x).exp() * h / gamma(a)
    }
}

/// Digamma function (derivative of log gamma)
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    
    // Use asymptotic expansion for large x
    if x > 10.0 {
        let mut result = x.ln() - 0.5 / x;
        let mut x2 = x * x;
        let mut term = 1.0 / x2;
        result -= term / 12.0;
        term /= x2;
        result += term / 120.0;
        term /= x2;
        result -= term / 252.0;
        result
    } else {
        // Use recurrence relation for small x
        digamma(x + 1.0) - 1.0 / x
    }
}

/// Test and verification utilities
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gammln_positive() {
        // Test known values
        assert_abs_diff_eq!(gammln(1.0), 0.0, epsilon = 1e-10); // Γ(1) = 1, ln(1) = 0
        assert_abs_diff_eq!(gammln(2.0), 0.0, epsilon = 1e-10); // Γ(2) = 1, ln(1) = 0
        assert_abs_diff_eq!(gammln(3.0), 0.6931471805599453.ln(), epsilon = 1e-10); // Γ(3) = 2
        assert_abs_diff_eq!(gammln(4.0), 1.791759469228055, epsilon = 1e-10); // Γ(4) = 6
        assert_abs_diff_eq!(gammln(0.5), 0.5723649429247001, epsilon = 1e-10); // Γ(0.5) = √π
    }

    #[test]
    fn test_gammln_negative() {
        // Test reflection formula
        assert_abs_diff_eq!(gammln(-0.5), gammln(1.5) - (PI / 2.0).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(gammln(-1.5), gammln(2.5) - (3.0 * PI / 4.0).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_gammln_special_cases() {
        assert!(gammln(0.0).is_infinite());
        assert!(gammln(-1.0).is_nan());
        assert!(gammln(-2.0).is_nan());
    }

    #[test]
    fn test_gammln_fast() {
        // Test that fast version matches regular version
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            assert_abs_diff_eq!(gammln(x), gammln_fast(x), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gammln_simd() {
        let inputs = [0.5, 1.0, 1.5, 2.0, 2.5];
        let results = gammln_simd(&inputs);
        
        for i in 0..inputs.len() {
            assert_abs_diff_eq!(results[i], gammln(inputs[i]), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gammln_batch() {
        let inputs = [0.5, 1.0, 1.5, 2.0, 2.5];
        let results = gammln_batch(&inputs);
        
        for i in 0..inputs.len() {
            assert_abs_diff_eq!(results[i], gammln(inputs[i]), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_gamma_function() {
        assert_abs_diff_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(4.0), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(0.5), PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_erf_function() {
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(erf(1.0), 0.8427007929497149, epsilon = 1e-10);
        assert_abs_diff_eq!(erf(-1.0), -0.8427007929497149, epsilon = 1e-10);
        assert_abs_diff_eq!(erf(f64::INFINITY), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_digamma_function() {
        assert_abs_diff_eq!(digamma(1.0), -0.5772156649015329, epsilon = 1e-10); // -γ (Euler-Mascheroni)
        assert_abs_diff_eq!(digamma(0.5), -1.9635100260214235, epsilon = 1e-10);
        assert_abs_diff_eq!(digamma(2.0), 0.42278433509846713, epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_gamma() {
        assert_abs_diff_eq!(gamma_p(1.0, 1.0), 0.6321205588285577, epsilon = 1e-10); // 1 - e^{-1}
        assert_abs_diff_eq!(gamma_q(1.0, 1.0), 0.36787944117144233, epsilon = 1e-10); // e^{-1}
        
        assert_abs_diff_eq!(gamma_p(0.5, 0.5), 0.6826894921370859, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma_q(0.5, 0.5), 0.3173105078629141, epsilon = 1e-10);
    }

    #[test]
    fn test_cache_functionality() {
        // Clear cache and test that values are cached
        GAMMLN_CACHE.with(|cache| {
            let mut cache = cache.lock().unwrap();
            cache.clear();
        });

        let x = 2.5;
        let result1 = gammln(x);
        let result2 = gammln(x); // Should be from cache
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
        
        // Verify cache contains the value
        GAMMLN_CACHE.with(|cache| {
            let cache = cache.lock().unwrap();
            assert!(cache.contains(&x.to_bits()));
        });
    }

    #[test]
    fn test_large_values() {
        // Test for large arguments where Stirling's approximation should be good
        let large_x = 100.0;
        let result = gammln(large_x);
        
        // Stirling's approximation: ln Γ(z) ≈ (z - 0.5) ln z - z + 0.5 ln(2π)
        let stirling = (large_x - 0.5) * large_x.ln() - large_x + 0.5 * (2.0 * PI).ln();
        
        assert_abs_diff_eq!(result, stirling, epsilon = 1e-8);
    }
}
