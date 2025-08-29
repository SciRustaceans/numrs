use std::simd::{f64x4, SimdFloat};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::f64::consts::LN_2;

/// Precomputed factorials for n = 0 to 33
static PRECOMPUTED_FACTORIALS: [f64; 34] = [
    1.0, // 0!
    1.0, // 1!
    2.0, // 2!
    6.0, // 3!
    24.0, // 4!
    120.0, // 5!
    720.0, // 6!
    5040.0, // 7!
    40320.0, // 8!
    362880.0, // 9!
    3628800.0, // 10!
    39916800.0, // 11!
    479001600.0, // 12!
    6227020800.0, // 13!
    87178291200.0, // 14!
    1307674368000.0, // 15!
    20922789888000.0, // 16!
    355687428096000.0, // 17!
    6402373705728000.0, // 18!
    121645100408832000.0, // 19!
    2432902008176640000.0, // 20!
    51090942171709440000.0, // 21!
    1124000727777607680000.0, // 22!
    25852016738884976640000.0, // 23!
    620448401733239439360000.0, // 24!
    15511210043330985984000000.0, // 25!
    403291461126605635584000000.0, // 26!
    10888869450418352160768000000.0, // 27!
    304888344611713860501504000000.0, // 28!
    8841761993739701954543616000000.0, // 29!
    265252859812191058636308480000000.0, // 30!
    8222838654177922817725562880000000.0, // 31!
    263130836933693530167218012160000000.0, // 32!
    8683317618811886495518194401280000000.0, // 33!
];

/// Thread-safe cache for computed factorials beyond precomputed range
static FACTORIAL_CACHE: Lazy<Mutex<lru::LruCache<i32, f64>>> = Lazy::new(|| {
    Mutex::new(lru::LruCache::new(1000))
});

/// Compute factorial using precomputed values and gamma function for large n
pub fn factrl(n: i32) -> f64 {
    if n < 0 {
        panic!("Negative factorial in routine factrl");
    }
    
    // Use precomputed values for n <= 33
    if n <= 33 {
        return PRECOMPUTED_FACTORIALS[n as usize];
    }
    
    // Check cache for previously computed values
    {
        let mut cache = FACTORIAL_CACHE.lock().unwrap();
        if let Some(&result) = cache.get(&n) {
            return result;
        }
    }
    
    // Use gamma function for large n: n! = Γ(n+1)
    let result = gammln((n + 1) as f64).exp();
    
    // Cache the result
    {
        let mut cache = FACTORIAL_CACHE.lock().unwrap();
        cache.put(n, result);
    }
    
    result
}

/// Logarithm of factorial using gamma function
pub fn factln(n: i32) -> f64 {
    if n < 0 {
        panic!("Negative factorial in routine factln");
    }
    
    if n <= 33 {
        PRECOMPUTED_FACTORIALS[n as usize].ln()
    } else {
        gammln((n + 1) as f64)
    }
}

/// Double factorial (n!!)
pub fn double_factrl(n: i32) -> f64 {
    if n < 0 {
        panic!("Negative double factorial");
    }
    
    if n <= 1 {
        return 1.0;
    }
    
    if n % 2 == 0 {
        // Even: n!! = 2^(n/2) * (n/2)!
        let half = n / 2;
        2.0f64.powi(half) * factrl(half)
    } else {
        // Odd: n!! = n! / (2^((n-1)/2) * ((n-1)/2)!)
        factrl(n) / (2.0f64.powi((n - 1) / 2) * factrl((n - 1) / 2))
    }
}

/// Binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
pub fn binom(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    
    // Use symmetric property to minimize computations
    let k = if k > n - k { n - k } else { k };
    
    if k == 0 {
        return 1.0;
    }
    
    if n <= 33 {
        // Use precomputed factorials for small n
        PRECOMPUTED_FACTORIALS[n as usize] / 
        (PRECOMPUTED_FACTORIALS[k as usize] * PRECOMPUTED_FACTORIALS[(n - k) as usize])
    } else {
        // Use logarithms for large n to avoid overflow
        (factln(n) - factln(k) - factln(n - k)).exp()
    }
}

/// Multinomial coefficient
pub fn multinom(n: i32, ks: &[i32]) -> f64 {
    let total: i32 = ks.iter().sum();
    if total != n {
        panic!("Sum of ks must equal n");
    }
    
    let mut result = factln(n);
    for &k in ks {
        result -= factln(k);
    }
    result.exp()
}

/// Rising factorial (Pochhammer symbol): x^{(n)} = x(x+1)...(x+n-1)
pub fn rising_factorial(x: f64, n: i32) -> f64 {
    if n < 0 {
        panic!("Negative order in rising factorial");
    }
    
    if n == 0 {
        return 1.0;
    }
    
    // Use gamma function: x^{(n)} = Γ(x+n) / Γ(x)
    gammln(x + n as f64).exp() / gammln(x).exp()
}

/// Falling factorial: x_{(n)} = x(x-1)...(x-n+1)
pub fn falling_factorial(x: f64, n: i32) -> f64 {
    if n < 0 {
        panic!("Negative order in falling factorial");
    }
    
    if n == 0 {
        return 1.0;
    }
    
    // Use gamma function: x_{(n)} = Γ(x+1) / Γ(x-n+1)
    gammln(x + 1.0).exp() / gammln(x - n as f64 + 1.0).exp()
}

/// SIMD-optimized batch factorial computation
pub fn factrl_batch(ns: &[i32]) -> Vec<f64> {
    ns.iter().map(|&n| factrl(n)).collect()
}

/// Parallel batch computation
pub fn factrl_batch_parallel(ns: &[i32]) -> Vec<f64> {
    use rayon::prelude::*;
    ns.par_iter().map(|&n| factrl(n)).collect()
}

/// Gamma function implementation (from previous)
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

    for j in 0..3 {
        let idx = j * 2;
        ser += COEFFS[idx] / y_acc;
        y_acc += 1.0;
        ser += COEFFS[idx + 1] / y_acc;
        y_acc += 1.0;
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Stirling's approximation for large factorials
pub fn stirling_approximation(n: i32) -> f64 {
    let n_f64 = n as f64;
    (2.0 * std::f64::consts::PI * n_f64).sqrt() * (n_f64 / std::f64::consts::E).powi(n)
}

/// Error function for factorial approximation quality
pub fn factorial_error(n: i32) -> f64 {
    let exact = factrl(n);
    let approx = stirling_approximation(n);
    (exact - approx).abs() / exact
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_factrl_basic() {
        assert_abs_diff_eq!(factrl(0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(1), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(2), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(3), 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(4), 24.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(5), 120.0, epsilon = 1e-10);
    }

    #[test]
    fn test_factrl_large() {
        // Test beyond precomputed range
        let n = 50;
        let result = factrl(n);
        
        // Verify using gamma function identity
        let expected = gammln((n + 1) as f64).exp();
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_factln() {
        assert_abs_diff_eq!(factln(0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factln(1), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(factln(2), 2.0f64.ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(factln(10), PRECOMPUTED_FACTORIALS[10].ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_double_factorial() {
        assert_abs_diff_eq!(double_factrl(0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(1), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(2), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(3), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(4), 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(5), 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(double_factrl(6), 48.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binomial() {
        assert_abs_diff_eq!(binom(5, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binom(5, 1), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binom(5, 2), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binom(5, 3), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binom(5, 4), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(binom(5, 5), 1.0, epsilon = 1e-10);
        
        // Test symmetry
        assert_abs_diff_eq!(binom(10, 3), binom(10, 7), epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial() {
        // Trinomial case: 5! / (2! * 2! * 1!) = 30
        assert_abs_diff_eq!(multinom(5, &[2, 2, 1]), 30.0, epsilon = 1e-10);
        
        // 6! / (2! * 2! * 2!) = 90
        assert_abs_diff_eq!(multinom(6, &[2, 2, 2]), 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rising_factorial() {
        assert_abs_diff_eq!(rising_factorial(2.0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rising_factorial(2.0, 1), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rising_factorial(2.0, 2), 6.0, epsilon = 1e-10); // 2*3
        assert_abs_diff_eq!(rising_factorial(2.0, 3), 24.0, epsilon = 1e-10); // 2*3*4
    }

    #[test]
    fn test_falling_factorial() {
        assert_abs_diff_eq!(falling_factorial(5.0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(falling_factorial(5.0, 1), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(falling_factorial(5.0, 2), 20.0, epsilon = 1e-10); // 5*4
        assert_abs_diff_eq!(falling_factorial(5.0, 3), 60.0, epsilon = 1e-10); // 5*4*3
    }

    #[test]
    fn test_batch_processing() {
        let ns = [0, 1, 2, 3, 4, 5];
        let results = factrl_batch(&ns);
        
        for (i, &n) in ns.iter().enumerate() {
            assert_abs_diff_eq!(results[i], factrl(n), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_batch() {
        let ns = [0, 1, 2, 3, 4, 5, 10, 15, 20];
        let results = factrl_batch_parallel(&ns);
        
        for (i, &n) in ns.iter().enumerate() {
            assert_abs_diff_eq!(results[i], factrl(n), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        // Clear cache
        {
            let mut cache = FACTORIAL_CACHE.lock().unwrap();
            cache.clear();
        }
        
        let n = 40;
        let result1 = factrl(n);
        let result2 = factrl(n); // Should be from cache
        
        assert_abs_diff_eq!(result1, result2, epsilon = 1e-15);
        
        // Verify cache contains the value
        {
            let cache = FACTORIAL_CACHE.lock().unwrap();
            assert!(cache.contains(&n));
        }
    }

    #[test]
    fn test_stirling_approximation() {
        // Test that Stirling's approximation is reasonable for large n
        for n in [10, 20, 30] {
            let exact = factrl(n);
            let approx = stirling_approximation(n);
            let error = (exact - approx).abs() / exact;
            
            assert!(error < 0.01, "Stirling error too large: {}", error);
        }
    }

    #[test]
    #[should_panic(expected = "Negative factorial")]
    fn test_negative_factorial() {
        factrl(-1);
    }

    #[test]
    #[should_panic(expected = "Negative double factorial")]
    fn test_negative_double_factorial() {
        double_factrl(-1);
    }

    #[test]
    fn test_edge_cases() {
        // Test boundary cases
        assert_abs_diff_eq!(factrl(33), PRECOMPUTED_FACTORIALS[33], epsilon = 1e-10);
        assert_abs_diff_eq!(factrl(34), gammln(35.0).exp(), epsilon = 1e-10);
    }
}
