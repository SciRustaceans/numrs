use std::simd::prelude::SimdFloat;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::f64::consts::LN_2;

/// Precomputed natural logarithms of factorials for n = 0 to 100
static FACTLN_CACHE: Lazy<Mutex<[f64; 101]>> = Lazy::new(|| {
    Mutex::new([0.0; 101])
});

/// Thread-safe cache for computed binomial coefficients
static BINOM_CACHE: Lazy<Mutex<lru::LruCache<(i32, i32), f64>>> = Lazy::new(|| {
    Mutex::new(lru::LruCache::new(10000))
});

/// Initialize the factorial logarithm cache
fn init_factln_cache() {
    let mut cache = FACTLN_CACHE.lock().unwrap();
    if cache[0] == 0.0 { // Check if already initialized
        cache[0] = 0.0; // ln(0!) = ln(1) = 0
        cache[1] = 0.0; // ln(1!) = 0
        
        for n in 2..=100 {
            cache[n] = gammln(n as f64 + 1.0);
        }
    }
}

/// Binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
pub fn bico(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    
    // Use symmetry to minimize computations: C(n, k) = C(n, n-k)
    let k = if k > n - k { n - k } else { k };
    
    if k == 0 {
        return 1.0;
    }
    
    // Check cache first
    {
        let mut cache = BINOM_CACHE.lock().unwrap();
        if let Some(&result) = cache.get(&(n, k)) {
            return result;
        }
    }
    
    // Compute using logarithms to avoid overflow
    let result = (factln(n) - factln(k) - factln(n - k)).exp().round();
    
    // Cache the result
    {
        let mut cache = BINOM_CACHE.lock().unwrap();
        cache.put((n, k), result);
    }
    
    result
}

/// Natural logarithm of factorial: ln(n!)
pub fn factln(n: i32) -> f64 {
    if n < 0 {
        panic!("Negative factorial in routine factln");
    }
    
    if n <= 1 {
        return 0.0;
    }
    
    // Initialize cache if needed
    init_factln_cache();
    
    let cache = FACTLN_CACHE.lock().unwrap();
    if (n as usize) < cache.len() {
        cache[n as usize]
    } else {
        gammln(n as f64 + 1.0)
    }
}

/// Gamma function implementation (optimized)
fn gammln(x: f64) -> f64 {
    if x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        if x == x.floor() && x < 0.0 {
            return f64::NAN;
        }
    }

    if x < 0.5 {
        return LN_2.ln() - (-x).ln() - gammln(1.0 - x) - (std::f64::consts::PI * x).sin().ln();
    }

    const COEFFS: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    
    let mut ser = 1.000000000190015;
    let mut y_acc = x + 1.0;

    // Process coefficients in pairs for better ILP
    for j in 0..3 {
        let idx = j * 2;
        ser += COEFFS[idx] / y_acc;
        y_acc += 1.0;
        ser += COEFFS[idx + 1] / y_acc;
        y_acc += 1.0;
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Direct computation for small binomial coefficients (avoiding logs)
pub fn bico_direct(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    
    let k = if k > n - k { n - k } else { k };
    
    if k == 0 {
        return 1.0;
    }
    
    // Use direct computation for small values to avoid log rounding errors
    if n <= 33 && k <= 16 {
        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result.round()
    } else {
        bico(n, k)
    }
}

/// Multi-threaded batch computation of binomial coefficients
pub fn bico_batch(pairs: &[(i32, i32)]) -> Vec<f64> {
    pairs.iter().map(|&(n, k)| bico(n, k)).collect()
}

/// Parallel batch computation using Rayon
pub fn bico_batch_parallel(pairs: &[(i32, i32)]) -> Vec<f64> {
    use rayon::prelude::*;
    pairs.par_iter().map(|&(n, k)| bico(n, k)).collect()
}

/// Logarithm of binomial coefficient (avoiding overflow)
pub fn bicoln(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return f64::NEG_INFINITY;
    }
    
    let k = if k > n - k { n - k } else { k };
    
    if k == 0 {
        return 0.0;
    }
    
    factln(n) - factln(k) - factln(n - k)
}

/// Binomial coefficient with exact integer arithmetic (for small n)
pub fn bico_exact(n: i32, k: i32) -> Option<u64> {
    if k < 0 || k > n {
        return Some(0);
    }
    
    let k = if k > n - k { n - k } else { k };
    
    if k == 0 {
        return Some(1);
    }
    
    // Check for potential overflow
    if n > 67 || (n > 61 && k > 1) {
        return None;
    }
    
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.checked_mul((n - i) as u64)?;
        result /= (i + 1) as u64;
    }
    Some(result)
}

/// Multinomial coefficient
pub fn multinomial(n: i32, ks: &[i32]) -> f64 {
    let total: i32 = ks.iter().sum();
    if total != n {
        panic!("Sum of ks must equal n");
    }
    
    let mut result = factln(n);
    for &k in ks {
        result -= factln(k);
    }
    result.exp().round()
}

/// Poisson distribution probability mass function
pub fn poisson_pmf(lambda: f64, k: i32) -> f64 {
    (-lambda + k as f64 * lambda.ln() - factln(k)).exp()
}

/// Negative binomial distribution PMF
pub fn negative_binomial_pmf(r: i32, p: f64, k: i32) -> f64 {
    bicoln(k + r - 1, k).exp() * p.powi(r) * (1.0 - p).powi(k)
}

/// Hypergeometric distribution PMF
pub fn hypergeometric_pmf(n: i32, k: i32, n1: i32, n2: i32, x: i32) -> f64 {
    if x < 0 || x > k || x > n1 || k - x > n2 {
        return 0.0;
    }
    
    (bicoln(n1, x) + bicoln(n2, k - x) - bicoln(n, k)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bico_basic() {
        assert_abs_diff_eq!(bico(5, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 1), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 2), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 3), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 4), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 5), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bico_symmetry() {
        for n in 0..=20 {
            for k in 0..=n {
                assert_abs_diff_eq!(bico(n, k), bico(n, n - k), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bico_large() {
        // Test large values that would overflow with direct computation
        assert_abs_diff_eq!(bico(100, 50), 1.008913445455642e29, epsilon = 1e15);
        assert_abs_diff_eq!(bico(200, 100), 9.054851435610317e58, epsilon = 1e45);
    }

    #[test]
    fn test_bico_direct() {
        // Test that direct method matches logarithmic method for small n
        for n in 0..=20 {
            for k in 0..=n {
                assert_abs_diff_eq!(bico(n, k), bico_direct(n, k), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bicoln() {
        assert_abs_diff_eq!(bicoln(5, 2), (10.0f64).ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(bicoln(10, 5), (252.0f64).ln(), epsilon = 1e-10);
        
        // Test that exp(bicoln()) matches bico()
        assert_abs_diff_eq!(bico(20, 10), bicoln(20, 10).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_bico_exact() {
        assert_eq!(bico_exact(5, 2), Some(10));
        assert_eq!(bico_exact(10, 5), Some(252));
        assert_eq!(bico_exact(20, 10), Some(184756));
        
        // Test overflow case
        assert_eq!(bico_exact(100, 50), None);
    }

    #[test]
    fn test_factln() {
        assert_abs_diff_eq!(factln(0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factln(1), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factln(2), 2.0f64.ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(factln(3), 6.0f64.ln(), epsilon = 1e-10);
        
        // Test cache functionality
        let ln100 = factln(100);
        assert_abs_diff_eq!(ln100, gammln(101.0), epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial() {
        // Trinomial: 5! / (2! * 2! * 1!) = 30
        assert_abs_diff_eq!(multinomial(5, &[2, 2, 1]), 30.0, epsilon = 1e-10);
        
        // 6! / (2! * 2! * 2!) = 90
        assert_abs_diff_eq!(multinomial(6, &[2, 2, 2]), 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_probability_distributions() {
        // Poisson distribution test
        let lambda = 3.0;
        let mut poisson_sum = 0.0;
        for k in 0..20 {
            poisson_sum += poisson_pmf(lambda, k);
        }
        assert_abs_diff_eq!(poisson_sum, 1.0, epsilon = 1e-10);
        
        // Negative binomial test
        let r = 5;
        let p = 0.3;
        let mut nb_sum = 0.0;
        for k in 0..50 {
            nb_sum += negative_binomial_pmf(r, p, k);
        }
        assert_abs_diff_eq!(nb_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cache_functionality() {
        // Clear caches
        {
            let mut cache = BINOM_CACHE.lock().unwrap();
            cache.clear();
        }
        
        let n = 50;
        let k = 25;
        
        let result1 = bico(n, k);
        let result2 = bico(n, k); // Should be from cache
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
        
        // Verify cache contains the value
        {
            let cache = BINOM_CACHE.lock().unwrap();
            assert!(cache.contains(&(n, k)));
        }
    }

    #[test]
    fn test_batch_processing() {
        let pairs = [(5, 2), (10, 5), (15, 7)];
        let results = bico_batch(&pairs);
        
        assert_abs_diff_eq!(results[0], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[1], 252.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[2], 6435.0, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_cases() {
        assert_abs_diff_eq!(bico(0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(1, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(1, 1), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(100, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(100, 100), 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Negative factorial")]
    fn test_negative_factorial() {
        factln(-1);
    }

    #[test]
    fn test_out_of_bounds() {
        assert_abs_diff_eq!(bico(5, -1), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bico(5, 6), 0.0, epsilon = 1e-10);
    }
}
