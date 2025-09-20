use std::sync::{Arc, Mutex};
use std::thread;

const ACC: i32 = 40;
const BIGNO: f64 = 1.0e10;
const BIGNI: f64 = 1.0e-10;

/// Computes the modified Bessel function of the first kind Iₙ(x) for integer order n ≥ 2
/// Uses Miller's algorithm with downward recurrence for numerical stability
/// Supports both f32 and f64 precision through generics
pub fn bessi<T>(n: i32, x: T) -> T
where
    T: BesselFloat,
{
    if n < 2 {
        panic!("bessi: Index n less than 2 in bessi");
    }

    if x == T::zero() {
        return T::zero();
    }

    let ax = x.abs();
    let tox = T::from_f64(2.0).unwrap() / ax;
    
    let mut bip = T::zero();
    let mut ans = T::zero();
    let mut bi = T::one();
    
    // Determine starting point for downward recurrence
    let m = 2 * (n + ((ACC * n) as f64).sqrt() as i32);
    
    for j in (1..=m).rev() {
        let bim = bip + T::from_i32(j).unwrap() * tox * bi;
        bip = bi;
        bi = bim;
        
        // Renormalize to prevent overflow
        if bi.abs() > T::from_f64(BIGNO).unwrap() {
            ans = ans * T::from_f64(BIGNI).unwrap();
            bi = bi * T::from_f64(BIGNI).unwrap();
            bip = bip * T::from_f64(BIGNI).unwrap();
        }
        
        if j == n {
            ans = bip;
        }
    }
    
    // Normalize using I₀(x)
    ans = ans * bessi0(x) / bi;
    
    // Handle sign for negative x and odd n
    if x < T::zero() && (n & 1) == 1 {
        -ans
    } else {
        ans
    }
}

/// Computes the modified Bessel function of the first kind I₀(x)
pub fn bessi0<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax < T::from_f64(3.75).unwrap() {
        let y = (x / T::from_f64(3.75).unwrap()).powi(2);
        polynomial_eval(
            y,
            &[
                T::from_f64(1.0).unwrap(),
                T::from_f64(3.5156229).unwrap(),
                T::from_f64(3.0899424).unwrap(),
                T::from_f64(1.2067492).unwrap(),
                T::from_f64(0.2659732).unwrap(),
                T::from_f64(0.360768e-1).unwrap(),
                T::from_f64(0.45813e-2).unwrap(),
            ],
        )
    } else {
        let y = T::from_f64(3.75).unwrap() / ax;
        polynomial_eval(
            y,
            &[
                T::from_f64(0.39894228).unwrap(),
                T::from_f64(0.1328592e-1).unwrap(),
                T::from_f64(0.225319e-2).unwrap(),
                T::from_f64(-0.157565e-2).unwrap(),
                T::from_f64(0.916281e-2).unwrap(),
                T::from_f64(-0.2057706e-1).unwrap(),
                T::from_f64(0.2635537e-1).unwrap(),
                T::from_f64(-0.1647633e-1).unwrap(),
                T::from_f64(0.392377e-2).unwrap(),
            ],
        ) * ax.exp() / ax.sqrt()
    }
}

/// Computes the modified Bessel function of the first kind I₁(x)
pub fn bessi1<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax <= T::from_f64(3.75).unwrap() {
        let y = (x / T::from_f64(3.75).unwrap()).powi(2);
        x * polynomial_eval(
            y,
            &[
                T::from_f64(0.5).unwrap(),
                T::from_f64(0.87890594).unwrap(),
                T::from_f64(0.51498869).unwrap(),
                T::from_f64(0.15084934).unwrap(),
                T::from_f64(0.2658733e-1).unwrap(),
                T::from_f64(0.301532e-2).unwrap(),
                T::from_f64(0.32411e-3).unwrap(),
            ],
        )
    } else {
        let y = T::from_f64(3.75).unwrap() / ax;
        
        let poly = polynomial_eval(
            y,
            &[
                T::from_f64(0.39894228).unwrap(),
                T::from_f64(-0.3988024e-1).unwrap(),
                T::from_f64(-0.362018e-2).unwrap(),
                T::from_f64(0.163801e-2).unwrap(),
                T::from_f64(-0.1031555e-1).unwrap(),
                T::from_f64(0.2282967e-1).unwrap(),
                T::from_f64(-0.2895312e-1).unwrap(),
                T::from_f64(0.1787654e-1).unwrap(),
                T::from_f64(-0.420059e-2).unwrap(),
            ],
        );
        
        let result = poly * ax.exp() / ax.sqrt();
        
        if x < T::zero() {
            -result
        } else {
            result
        }
    }
}

/// Multithreaded computation of modified Bessel functions Iₙ(x) for multiple orders ≥ 2
pub fn bessi_multithreaded<T>(orders: &[i32], x: T, num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    // Filter out orders < 2 and handle errors
    let valid_orders: Vec<i32> = orders.iter()
        .filter(|&&n| n >= 2)
        .cloned()
        .collect();
    
    if valid_orders.len() != orders.len() {
        panic!("bessi_multithreaded: all orders must be ≥ 2");
    }

    let orders_chunks: Vec<Vec<i32>> = valid_orders
        .chunks((valid_orders.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let x_arc = Arc::new(x);
    let results = Arc::new(Mutex::new(vec![T::zero(); valid_orders.len()]));

    let mut handles = vec![];

    for (thread_idx, chunk) in orders_chunks.into_iter().enumerate() {
        let x_ref = Arc::clone(&x_arc);
        let results_ref = Arc::clone(&results);
        let start_idx = thread_idx * chunk.len();

        handles.push(thread::spawn(move || {
            for (local_idx, &n) in chunk.iter().enumerate() {
                let global_idx = start_idx + local_idx;
                let result = bessi(n, *x_ref);
                let mut results_lock = results_ref.lock().unwrap();
                results_lock[global_idx] = result;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(results)
        .unwrap()
        .into_inner()
        .unwrap()
}

/// Computes Iₙ(x) for a range of orders n ≥ 2 using optimized Miller's algorithm
pub fn bessi_range<T>(n_start: i32, n_end: i32, x: T) -> Vec<T>
where
    T: BesselFloat,
{
    if n_start < 2 || n_end < n_start {
        panic!("bessi_range: n_start must be ≥ 2 and n_end ≥ n_start");
    }

    if x == T::zero() {
        return vec![T::zero(); (n_end - n_start + 1) as usize];
    }

    let ax = x.abs();
    let tox = T::from_f64(2.0).unwrap() / ax;
    
    let mut results = Vec::with_capacity((n_end - n_start + 1) as usize);
    
    // Use Miller's algorithm for the highest order needed
    let highest_n = n_end;
    let m = 2 * (highest_n + ((ACC * highest_n) as f64).sqrt() as i32);
    
    let mut bip = T::zero();
    let mut bi = T::one();
    let mut stored_values = vec![T::zero(); (highest_n + 1) as usize];
    
    for j in (1..=m).rev() {
        let bim = bip + T::from_i32(j).unwrap() * tox * bi;
        bip = bi;
        bi = bim;
        
        // Renormalize to prevent overflow
        if bi.abs() > T::from_f64(BIGNO).unwrap() {
            bi = bi * T::from_f64(BIGNI).unwrap();
            bip = bip * T::from_f64(BIGNI).unwrap();
            for value in &mut stored_values {
                *value = *value * T::from_f64(BIGNI).unwrap();
            }
        }
        
        if j <= highest_n {
            stored_values[j as usize] = bip;
        }
    }
    
    // Normalize using I₀(x)
    let normalization = bessi0(x) / bi;
    
    // Extract the requested range
    for n in n_start..=n_end {
        let mut ans = stored_values[n as usize] * normalization;
        
        // Handle sign for negative x and odd n
        if x < T::zero() && (n & 1) == 1 {
            ans = -ans;
        }
        
        results.push(ans);
    }
    
    results
}

/// Computes Iₙ(x) for multiple x values using multithreading
pub fn bessi_multiple_x<T>(n: i32, x_values: &[T], num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    if n < 2 {
        panic!("bessi_multiple_x: n must be ≥ 2");
    }

    let x_chunks: Vec<Vec<T>> = x_values
        .chunks((x_values.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let results = Arc::new(Mutex::new(vec![T::zero(); x_values.len()]));

    let mut handles = vec![];

    for (thread_idx, chunk) in x_chunks.into_iter().enumerate() {
        let results_ref = Arc::clone(&results);
        let start_idx = thread_idx * chunk.len();

        handles.push(thread::spawn(move || {
            for (local_idx, x) in chunk.iter().enumerate() {
                let global_idx = start_idx + local_idx;
                let result = bessi(n, *x);
                let mut results_lock = results_ref.lock().unwrap();
                results_lock[global_idx] = result;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(results)
        .unwrap()
        .into_inner()
        .unwrap()
}

/// Helper trait for Bessel function floating point operations
pub trait BesselFloat: 
    Copy + 
    PartialOrd + 
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + 
    std::ops::Div<Output = Self> + 
    std::ops::Neg<Output = Self> +
    FromPrimitive
{
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

/// Trait for converting from primitive types
pub trait FromPrimitive {
    fn from_i32(n: i32) -> Option<Self> where Self: Sized;
    fn from_f64(n: f64) -> Option<Self> where Self: Sized;
}

impl FromPrimitive for f32 {
    fn from_i32(n: i32) -> Option<Self> { Some(n as f32) }
    fn from_f64(n: f64) -> Option<Self> { Some(n as f32) }
}

impl FromPrimitive for f64 {
    fn from_i32(n: i32) -> Option<Self> { Some(n as f64) }
    fn from_f64(n: f64) -> Option<Self> { Some(n) }
}

/// Evaluates a polynomial using Horner's method
fn polynomial_eval<T>(x: T, coeffs: &[T]) -> T
where
    T: BesselFloat + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let mut result = T::zero();
    for &coeff in coeffs.iter().rev() {
        result = result * x + coeff;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bessi_basic() {
        // Test basic functionality for n ≥ 2
        assert_abs_diff_eq!(bessi(2, 1.0_f64), 0.135747, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(3, 1.0_f64), 0.022168, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(4, 1.0_f64), 0.002737, epsilon = 1e-6);
        
        assert_abs_diff_eq!(bessi(2, 2.0_f64), 0.688948, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(3, 2.0_f64), 0.212740, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi_zero_x() {
        // Iₙ(0) = 0 for n ≥ 1
        assert_eq!(bessi(2, 0.0_f64), 0.0);
        assert_eq!(bessi(3, 0.0_f64), 0.0);
        assert_eq!(bessi(4, 0.0_f64), 0.0);
    }

    #[test]
    #[should_panic(expected = "bessi: Index n less than 2 in bessi")]
    fn test_bessi_n_less_than_2() {
        bessi(1, 1.0_f64);
    }

    #[test]
    fn test_bessi_negative_x() {
        // Test that Iₙ(-x) = (-1)ⁿ Iₙ(x)
        assert_abs_diff_eq!(bessi(2, -1.0_f64), bessi(2, 1.0_f64), epsilon = 1e-12); // even n
        assert_abs_diff_eq!(bessi(3, -1.0_f64), -bessi(3, 1.0_f64), epsilon = 1e-12); // odd n
        assert_abs_diff_eq!(bessi(4, -2.0_f64), bessi(4, 2.0_f64), epsilon = 1e-12); // even n
        assert_abs_diff_eq!(bessi(5, -2.0_f64), -bessi(5, 2.0_f64), epsilon = 1e-12); // odd n
    }

    #[test]
    fn test_bessi_recurrence_relation() {
        // Test that the recurrence relation holds: I_{n-1}(x) - I_{n+1}(x) = (2n/x) I_n(x)
        let x = 3.0_f64;
        
        let i2 = bessi(2, x);
        let i3 = bessi(3, x);
        let i4 = bessi(4, x);
        
        // Test recurrence for n=3: I₂ - I₄ = (6/x) I₃
        let recurrence_value = i2 - i4;
        let expected = (6.0 / x) * i3;
        assert_abs_diff_eq!(recurrence_value, expected, epsilon = 1e-12);
        
        // Test another recurrence
        let i5 = bessi(5, x);
        let recurrence_value2 = i3 - i5;
        let expected2 = (8.0 / x) * i4;
        assert_abs_diff_eq!(recurrence_value2, expected2, epsilon = 1e-12);
    }

    #[test]
    fn test_bessi_multithreaded_basic() {
        let orders = vec![2, 3, 4, 5];
        let x = 2.0_f64;
        
        // Single-threaded reference
        let single_threaded: Vec<f64> = orders.iter().map(|&n| bessi(n, x)).collect();
        
        // Multi-threaded
        let multi_threaded = bessi_multithreaded(&orders, x, 2);
        
        for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st, *mt, epsilon = 1e-12, "Mismatch at order {}", orders[i]);
        }
    }

    #[test]
    #[should_panic(expected = "all orders must be ≥ 2")]
    fn test_bessi_multithreaded_invalid_orders() {
        let orders = vec![1, 2, 3]; // Contains n=1 which is invalid
        bessi_multithreaded(&orders, 2.0_f64, 2);
    }

    #[test]
    fn test_bessi_range() {
        let n_start = 2;
        let n_end = 5;
        let x = 1.5_f64;
        
        let range_results = bessi_range(n_start, n_end, x);
        
        assert_eq!(range_results.len(), 4);
        assert_abs_diff_eq!(range_results[0], bessi(2, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[1], bessi(3, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[2], bessi(4, x), epsilon = 1e-12);
        assert_abs_diff_eq!(range_results[3], bessi(5, x), epsilon = 1e-12);
    }

    #[test]
    fn test_bessi_multiple_x() {
        let n = 3;
        let x_values = vec![1.0_f64, 2.0, 3.0, 4.0];
        
        let multiple_x_results = bessi_multiple_x(n, &x_values, 2);
        
        assert_eq!(multiple_x_results.len(), 4);
        assert_abs_diff_eq!(multiple_x_results[0], bessi(n, 1.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[1], bessi(n, 2.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[2], bessi(n, 3.0), epsilon = 1e-12);
        assert_abs_diff_eq!(multiple_x_results[3], bessi(n, 4.0), epsilon = 1e-12);
    }

    #[test]
    fn test_bessi_large_orders() {
        // Test with larger orders
        assert_abs_diff_eq!(bessi(10, 5.0_f64), 0.001473, epsilon = 1e-6);
        assert_abs_diff_eq!(bessi(20, 10.0_f64), 0.000391, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi_large_x() {
        // Test with large x values - Iₙ(x) grows exponentially
        let x = 10.0_f64;
        assert_abs_diff_eq!(bessi(2, x), 2281.518, epsilon = 1e-3);
        assert_abs_diff_eq!(bessi(3, x), 2670.988, epsilon = 1e-3);
    }

    #[test]
    fn test_bessi_renormalization() {
        // Test that the renormalization works for cases that would overflow
        // Use a large x value that would cause overflow without renormalization
        let x = 20.0_f64;
        let n = 10;
        
        // This should not panic and should give a reasonable result
        let result = bessi(n, x);
        assert!(result > 0.0 && result < 1e10);
    }

    #[test]
    fn test_bessi_precision_consistency() {
        // Test consistency between f32 and f64
        let test_cases = [(2, 1.0), (3, 2.0), (4, 3.0)];
        
        for &(n, x) in &test_cases {
            let f32_result = bessi(n, x as f32);
            let f64_result = bessi(n, x as f64);
            
            assert_abs_diff_eq!(f32_result as f64, f64_result, epsilon = 1e-6,
                "Precision mismatch for I_{}({})", n, x);
        }
    }

    #[test]
    fn test_bessi_monotonicity_with_n() {
        // Test that Iₙ(x) decreases with increasing n for fixed x
        let x = 2.0_f64;
        let i2 = bessi(2, x);
        let i3 = bessi(3, x);
        let i4 = bessi(4, x);
        let i5 = bessi(5, x);
        
        assert!(i2 > i3 && i3 > i4 && i4 > i5,
            "Iₙ(x) should be decreasing with n for fixed x");
    }

    #[test]
    fn test_bessi_growth_with_x() {
        // Test that Iₙ(x) grows with increasing x for fixed n
        let n = 3;
        let i3_1 = bessi(n, 1.0_f64);
        let i3_2 = bessi(n, 2.0_f64);
        let i3_5 = bessi(n, 5.0_f64);
        
        assert!(i3_1 < i3_2 && i3_2 < i3_5,
            "Iₙ(x) should grow with increasing x for fixed n");
    }

    #[test]
    fn test_bessi_accuracy_against_known() {
        // Test against known mathematical values
        // I₂(1) should satisfy certain mathematical relationships
        let i2_1 = bessi(2, 1.0_f64);
        
        // Test derivative relationship near x=1
        let dx = 1e-8;
        let derivative_approx = (bessi(2, 1.0 + dx) - i2_1) / dx;
        
        // Theoretical derivative: dI₂/dx = I₁(x) - (3/x)I₂(x) + I₃(x)
        let i1_1 = bessi1(1.0);
        let i3_1 = bessi(3, 1.0);
        let theoretical_derivative = i1_1 - 3.0 * i2_1 + i3_1;
        
        assert_abs_diff_eq!(derivative_approx, theoretical_derivative, epsilon = 1e-6);
    }

    #[test]
    fn test_bessi_very_small_x() {
        // Test with very small x values
        let small_x = 1e-10_f64;
        assert_abs_diff_eq!(bessi(2, small_x), 1.25e-21, epsilon = 1e-31); // I₂(x) ~ x²/8 for small x
        assert_abs_diff_eq!(bessi(3, small_x), 1.04167e-32, epsilon = 1e-42); // I₃(x) ~ x³/48 for small x
    }
}
