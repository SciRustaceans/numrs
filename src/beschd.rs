use ndarray::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

const NUSE1: usize = 5;
const NUSE2: usize = 5;

/// Chebyshev polynomial evaluation for coefficients in range [a, b]
pub fn chebev<T>(a: T, b: T, c: &ArrayView1<T>, m: usize, x: T) -> T
where
    T: BesselFloat,
{
    if (x - a) * (x - b) > T::zero() {
        panic!("chebev: x not in range [a, b]");
    }

    let y = (T::from_f64(2.0).unwrap() * x - a - b) / (b - a);
    let y2 = T::from_f64(2.0).unwrap() * y;
    
    let mut d = T::zero();
    let mut dd = T::zero();
    
    for j in (1..m).rev() {
        let temp = d;
        d = y2 * d - dd + c[j];
        dd = temp;
    }
    
    y * d - dd + T::from_f64(0.5).unwrap() * c[0]
}

/// Computes Gamma function related quantities using Chebyshev approximations
pub fn beschd<T>(x: T) -> (T, T, T, T)
where
    T: BesselFloat,
{
    // Chebyshev coefficients for gam1 approximation using ndarray
    let c1 = Array1::from_vec(vec![
        T::from_f64(-1.142022680371168e0).unwrap(),
        T::from_f64(6.5165112670737e-3).unwrap(),
        T::from_f64(3.087090173086e-4).unwrap(),
        T::from_f64(-3.4706269649e-6).unwrap(),
        T::from_f64(6.9437664e-9).unwrap(),
        T::from_f64(3.67795e-11).unwrap(),
        T::from_f64(-1.356e-13).unwrap(),
    ]);

    // Chebyshev coefficients for gam2 approximation using ndarray
    let c2 = Array1::from_vec(vec![
        T::from_f64(1.843740587300905e0).unwrap(),
        T::from_f64(-7.6852840844867e-2).unwrap(),
        T::from_f64(1.2719271366546e-3).unwrap(),
        T::from_f64(-4.9717367042e-6).unwrap(),
        T::from_f64(-3.31261198e-8).unwrap(),
        T::from_f64(2.423096e-10).unwrap(),
        T::from_f64(-1.703e-13).unwrap(),
        T::from_f64(-1.49e-15).unwrap(),
    ]);

    let xx = T::from_f64(8.0).unwrap() * x * x - T::one();
    
    let gam1 = chebev(T::from_f64(-1.0).unwrap(), T::one(), &c1.view(), NUSE1, xx);
    let gam2 = chebev(T::from_f64(-1.0).unwrap(), T::one(), &c2.view(), NUSE2, xx);
    
    let gampl = gam2 - x * gam1;
    let gammi = gam2 + x * gam1;
    
    (gam1, gam2, gampl, gammi)
}

/// Alternative implementation using precise Gamma function calculations
pub fn beschd_precise<T>(x: T) -> (T, T, T, T)
where
    T: BesselFloat,
{
    // For small x, use series expansion
    if x.abs() < T::from_f64(0.1).unwrap() {
        return beschd_series(x);
    }
    
    // For larger x, use asymptotic expansion or Lanczos approximation
    beschd_asymptotic(x)
}

/// Series expansion for small x
fn beschd_series<T>(x: T) -> (T, T, T, T)
where
    T: BesselFloat,
{
    let x2 = x * x;
    let x4 = x2 * x2;
    
    // Series expansion coefficients for small x
    let gam1 = T::one() + x * (T::from_f64(0.57721566).unwrap() 
        + x * (T::from_f64(0.98905599).unwrap() 
        + x * (T::from_f64(0.90747908).unwrap())));
    
    let gam2 = T::one() / x - T::from_f64(0.57721566).unwrap() 
        + x * (T::from_f64(0.98905599).unwrap() 
        - x * (T::from_f64(0.90747908).unwrap()));
    
    let gampl = gam2 - x * gam1;
    let gammi = gam2 + x * gam1;
    
    (gam1, gam2, gampl, gammi)
}

/// Asymptotic expansion for larger x
fn beschd_asymptotic<T>(x: T) -> (T, T, T, T)
where
    T: BesselFloat,
{
    // Lanczos approximation for Gamma function
    let g = lanczos_gamma(x + T::one());
    let g_minus = lanczos_gamma(T::one() - x);
    
    let gam1 = (g - g_minus) / (T::from_f64(2.0).unwrap() * x);
    let gam2 = (g + g_minus) / T::from_f64(2.0).unwrap();
    
    let gampl = gam2 - x * gam1;
    let gammi = gam2 + x * gam1;
    
    (gam1, gam2, gampl, gammi)
}

/// Lanczos approximation for Gamma function
fn lanczos_gamma<T>(z: T) -> T
where
    T: BesselFloat + std::ops::SubAssign + std::ops::AddAssign,
{
    // Lanczos coefficients using ndarray
    let p = Array1::from_vec(vec![
        T::from_f64(676.5203681218851).unwrap(),
        T::from_f64(-1259.1392167224028).unwrap(),
        T::from_f64(771.32342877765313).unwrap(),
        T::from_f64(-176.61502916214059).unwrap(),
        T::from_f64(12.507343278686905).unwrap(),
        T::from_f64(-0.13857109526572012).unwrap(),
        T::from_f64(9.9843695780195716e-6).unwrap(),
        T::from_f64(1.5056327351493116e-7).unwrap(),
    ]);
    
    let mut x = z;
    if x < T::from_f64(0.5).unwrap() {
        // Reflection formula - use only T operations
        let pi = T::from_f64(std::f64::consts::PI).unwrap();
        return pi / ((pi * x).sin() * lanczos_gamma(T::one() - x));
    }
    
    x -= T::one();
    let mut a = T::from_f64(0.99999999999980993).unwrap();
    
    for (i, &p_val) in p.iter().enumerate() {
        a += p_val / (x + T::from_i32(i as i32 + 1).unwrap());
    }
    
    let t = x + T::from_f64(7.5).unwrap();
    let two_pi = T::from_f64(2.0 * std::f64::consts::PI).unwrap();
    two_pi.sqrt() * t.powf(x + T::from_f64(0.5).unwrap()) * (-t).exp() * a
}

/// Multithreaded computation of beschd for multiple x values using ndarray
pub fn beschd_multithreaded<T>(x_values: &ArrayView1<T>, num_threads: usize) -> Array1<(T, T, T, T)>
where
    T: BesselFloat + Send + Sync + Clone + 'static + std::fmt::Debug,
{
    let chunk_size = (x_values.len() + num_threads - 1) / num_threads;
    let chunks: Vec<Array1<T>> = x_values
        .axis_chunks_iter(ndarray::Axis(0), chunk_size)
        .map(|chunk| chunk.to_owned())
        .collect();

    let results = Arc::new(Mutex::new(Vec::with_capacity(x_values.len())));

    let mut handles = vec![];

    for chunk in chunks {
        let results_ref = Arc::clone(&results);
        
        handles.push(thread::spawn(move || {
            let chunk_results: Vec<(T, T, T, T)> = chunk.iter()
                .map(|&x| beschd(x))
                .collect();
            
            let mut results_lock = results_ref.lock().unwrap();
            results_lock.extend(chunk_results);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = Arc::try_unwrap(results)
        .unwrap()
        .into_inner()
        .unwrap();
    
    Array1::from_vec(final_results)
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
    std::ops::AddAssign +
    std::ops::SubAssign +
    FromPrimitive
{
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn powf(self, n: Self) -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn powf(self, n: Self) -> Self { self.powf(n) }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn powf(self, n: Self) -> Self { self.powf(n) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    #[test]
    fn test_chebev_basic() {
        // Test Chebyshev evaluation with simple polynomial
        let coeffs = arr1(&[1.0_f64, 2.0, 3.0]);
        let result = chebev(-1.0, 1.0, &coeffs.view(), 3, 0.0);
        
        // T₀(0)=1, T₁(0)=0, T₂(0)=-1
        // 1*1 + 2*0 + 3*(-1) = -2
        assert_abs_diff_eq!(result, -2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chebev_range() {
        // Test Chebyshev evaluation at boundaries
        let coeffs = arr1(&[1.0_f64, 1.0]);
        let result_min = chebev(-1.0, 1.0, &coeffs.view(), 2, -1.0);
        let result_max = chebev(-1.0, 1.0, &coeffs.view(), 2, 1.0);
        
        assert_abs_diff_eq!(result_min, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result_max, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "x not in range")]
    fn test_chebev_out_of_range() {
        let coeffs = arr1(&[1.0_f64]);
        chebev(-1.0, 1.0, &coeffs.view(), 1, 2.0);
    }

    #[test]
    fn test_beschd_small_x() {
        // Test for small x values
        let x = 0.1_f64;
        let (gam1, gam2, gampl, gammi) = beschd(x);
        
        // Verify basic properties
        assert!(gam1 > 0.0);
        assert!(gam2 > 0.0);
        assert_abs_diff_eq!(gampl, gam2 - x * gam1, epsilon = 1e-12);
        assert_abs_diff_eq!(gammi, gam2 + x * gam1, epsilon = 1e-12);
    }

    #[test]
    fn test_beschd_multithreaded_consistency() {
        let x_values = arr1(&[0.1_f64, 0.2, 0.3, 0.4, 0.5]);
        
        // Single-threaded reference
        let single_threaded: Vec<_> = x_values.iter()
            .map(|&x| beschd(x))
            .collect();
        
        // Multi-threaded
        let multi_threaded = beschd_multithreaded(&x_values.view(), 2);
        
        for i in 0..x_values.len() {
            let st = single_threaded[i];
            let mt = multi_threaded[i];
            
            assert_abs_diff_eq!(st.0, mt.0, epsilon = 1e-12);
            assert_abs_diff_eq!(st.1, mt.1, epsilon = 1e-12);
            assert_abs_diff_eq!(st.2, mt.2, epsilon = 1e-12);
            assert_abs_diff_eq!(st.3, mt.3, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_beschd_special_cases() {
        // Test special case x = 0
        let x = 0.0_f64;
        let (gam1, gam2, gampl, gammi) = beschd(x);
        
        // At x=0, gam1 should be finite, gam2 should be the limit
        assert!(gam1.is_finite());
        assert!(gam2.is_finite());
        assert_abs_diff_eq!(gampl, gam2, epsilon = 1e-12);
        assert_abs_diff_eq!(gammi, gam2, epsilon = 1e-12);
    }

    #[test]
    fn test_beschd_consistency() {
        // Test that the relationships always hold
        for &x in &[0.1, 0.5, 1.0, 2.0] {
            let (gam1, gam2, gampl, gammi) = beschd(x);
            
            assert_abs_diff_eq!(gampl, gam2 - x * gam1, epsilon = 1e-12);
            assert_abs_diff_eq!(gammi, gam2 + x * gam1, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_beschd_precision_consistency() {
        // Test consistency between f32 and f64
        let x = 0.3_f64;
        
        let (gam1_32, gam2_32, gampl_32, gammi_32) = beschd(x as f32);
        let (gam1_64, gam2_64, gampl_64, gammi_64) = beschd(x as f64);
        
        assert_abs_diff_eq!(gam1_32 as f64, gam1_64, epsilon = 1e-6);
        assert_abs_diff_eq!(gam2_32 as f64, gam2_64, epsilon = 1e-6);
        assert_abs_diff_eq!(gampl_32 as f64, gampl_64, epsilon = 1e-6);
        assert_abs_diff_eq!(gammi_32 as f64, gammi_64, epsilon = 1e-6);
    }

    #[test]
    fn test_lanczos_gamma() {
        // Test Lanczos approximation for Gamma function
        assert_abs_diff_eq!(lanczos_gamma(1.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lanczos_gamma(2.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lanczos_gamma(3.0_f64), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lanczos_gamma(4.0_f64), 6.0, epsilon = 1e-12);
        
        // Test reflection formula
        assert_abs_diff_eq!(lanczos_gamma(0.5_f64), std::f64::consts::PI.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_beschd_against_gamma_function() {
        // Test that beschd values are consistent with Gamma function properties
        let x = 0.25_f64;
        let (gam1, gam2, gampl, gammi) = beschd_precise(x);
        
        // Theoretical values using Gamma function
        let gamma_1_plus_x = lanczos_gamma(1.0 + x);
        let gamma_1_minus_x = lanczos_gamma(1.0 - x);
        
        let gam1_theoretical = (gamma_1_plus_x - gamma_1_minus_x) / (2.0 * x);
        let gam2_theoretical = (gamma_1_plus_x + gamma_1_minus_x) / 2.0;
        
        assert_abs_diff_eq!(gam1, gam1_theoretical, epsilon = 1e-10);
        assert_abs_diff_eq!(gam2, gam2_theoretical, epsilon = 1e-10);
    }
}
