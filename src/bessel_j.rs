use std::f64::consts;
use std::sync::{Arc, Mutex};
use std::thread;

const ACC: f64 = 40.0;
const BIGNO: f64 = 1.0e10;
const BIGNI: f64 = 1.0e-10;

/// Computes the Bessel function of the first kind Jₙ(x) for integer order n
/// Supports both f32 and f64 precision through generics
pub fn bessj<T>(n: i32, x: T) -> T
where
    T: BesselFloat,
{
    if n < 2 {
        panic!("Index n less than 2 in bessj");
    }

    let ax = x.abs();
    if ax == T::zero() {
        return T::zero();
    }

    if ax > T::from_i32(n).unwrap() {
        // Forward recurrence for large x
        let tox = T::from_f64(2.0).unwrap() / ax;
        let mut bjm = bessj0(ax);
        let mut bj = bessj1(ax);
        
        for j in 1..n {
            let j_val = T::from_i32(j).unwrap();
            let bjp = j_val * tox * bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        
        if x < T::zero() && (n & 1) == 1 {
            -bj
        } else {
            bj
        }
    } else {
        // Backward recurrence for small x
        let tox = T::from_f64(2.0).unwrap() / ax;
        let m = 2 * ((n + (ACC * n as f64).sqrt() as i32) / 2);
        
        let mut jsum = false;
        let mut bjp = T::zero();
        let mut ans = T::zero();
        let mut sum = T::zero();
        let mut bj = T::one();
        
        for j in (1..=m).rev() {
            let j_val = T::from_i32(j).unwrap();
            let bjm = j_val * tox * bj - bjp;
            bjp = bj;
            bj = bjm;
            
            if bj.abs() > T::from_f64(BIGNO).unwrap() {
                bj = bj * T::from_f64(BIGNI).unwrap();
                bjp = bjp * T::from_f64(BIGNI).unwrap();
                ans = ans * T::from_f64(BIGNI).unwrap();
                sum = sum * T::from_f64(BIGNI).unwrap();
            }
            
            if jsum {
                sum = sum + bj;
            }
            jsum = !jsum;
            
            if j == n {
                ans = bjp;
            }
        }
        
        sum = T::from_f64(2.0).unwrap() * sum - bj;
        let result = ans / sum;
        
        if x < T::zero() && (n & 1) == 1 {
            -result
        } else {
            result
        }
    }
}

/// Bessel function J₀(x) for f32 and f64
pub fn bessj0<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax < T::from_f64(8.0).unwrap() {
        let y = x * x;
        polynomial_eval(
            y,
            &[
                T::from_f64(-1.849_881_010_943_856_4e-3).unwrap(),
                T::from_f64(2.429_354_037_074_953_7e-1).unwrap(),
                T::from_f64(-4.908_139_957_392_126_5).unwrap(),
                T::from_f64(1.829_144_227_360_681_8e1).unwrap(),
                T::from_f64(-1.578_502_603_121_069_8e1).unwrap(),
            ],
        ) / polynomial_eval(
            y,
            &[
                T::from_f64(1.0).unwrap(),
                T::from_f64(4.126_898_326_313_704_5).unwrap(),
                T::from_f64(4.391_743_674_623_052_5).unwrap(),
                T::from_f64(1.385_532_355_036_583_8).unwrap(),
                T::from_f64(1.127_394_009_007_893_8e-1).unwrap(),
            ],
        )
    } else {
        let z = T::from_f64(8.0).unwrap() / ax;
        let y = z * z;
        let xx = ax - T::from_f64(consts::FRAC_PI_4).unwrap();
        
        let p0 = polynomial_eval(
            y,
            &[
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
            ],
        );
        
        let q0 = polynomial_eval(
            y,
            &[
                T::from_f64(0.398_942_279_2).unwrap(),
                T::from_f64(1.328_592_e-2).unwrap(),
                T::from_f64(2.253_187_e-3).unwrap(),
                T::from_f64(-1.575_653_e-3).unwrap(),
                T::from_f64(9.162_808_e-4).unwrap(),
                T::from_f64(-2.058_616_e-4).unwrap(),
                T::from_f64(2.464_680_e-5).unwrap(),
                T::from_f64(-1.226_625_e-6).unwrap(),
            ],
        );
        
        (T::from_f64(consts::FRAC_2_PI).unwrap().sqrt() / ax.sqrt())
            * (polynomial_eval(y, &[p0]) * xx.cos() - z * polynomial_eval(y, &[q0]) * xx.sin())
    }
}

/// Bessel function J₁(x) for f32 and f64
pub fn bessj1<T>(x: T) -> T
where
    T: BesselFloat,
{
    let ax = x.abs();
    
    if ax < T::from_f64(8.0).unwrap() {
        let y = x * x;
        x * polynomial_eval(
            y,
            &[
                T::from_f64(2.497_577_644_854_559_7e-3).unwrap(),
                T::from_f64(-2.177_540_137_813_397_2e-1).unwrap(),
                T::from_f64(3.029_626_247_966_687_7).unwrap(),
                T::from_f64(-9.228_661_324_469_804).unwrap(),
                T::from_f64(4.144_211_828_570_529).unwrap(),
            ],
        ) / polynomial_eval(
            y,
            &[
                T::from_f64(1.0).unwrap(),
                T::from_f64(4.272_916_361_347_971).unwrap(),
                T::from_f64(4.931_725_949_128_406).unwrap(),
                T::from_f64(1.689_047_419_900_923_2).unwrap(),
                T::from_f64(1.356_802_928_718_904_3e-1).unwrap(),
            ],
        )
    } else {
        let z = T::from_f64(8.0).unwrap() / ax;
        let y = z * z;
        let xx = ax - T::from_f64(3.0 * consts::FRAC_PI_4).unwrap();
        
        let p1 = polynomial_eval(
            y,
            &[
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
                T::from_f64(0.0).unwrap(),
            ],
        );
        
        let q1 = polynomial_eval(
            y,
            &[
                T::from_f64(0.398_942_279_1).unwrap(),
                T::from_f64(-3.988_024_e-2).unwrap(),
                T::from_f64(-3.620_182_e-3).unwrap(),
                T::from_f64(1.638_014_e-3).unwrap(),
                T::from_f64(-1.031_555_e-3).unwrap(),
                T::from_f64(2.282_698_e-4).unwrap(),
                T::from_f64(-2.895_106_e-5).unwrap(),
                T::from_f64(1.787_020_e-6).unwrap(),
            ],
        );
        
        let sign = if x < T::zero() { -T::one() } else { T::one() };
        sign * (T::from_f64(consts::FRAC_2_PI).unwrap().sqrt() / ax.sqrt())
            * (polynomial_eval(y, &[p1]) * xx.cos() - z * polynomial_eval(y, &[q1]) * xx.sin())
    }
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
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn cos(self) -> Self { self.cos() }
    fn sin(self) -> Self { self.sin() }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn cos(self) -> Self { self.cos() }
    fn sin(self) -> Self { self.sin() }
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

/// Multithreaded Bessel function computation for multiple orders
pub fn bessj_multithreaded<T>(orders: &[i32], x: T, num_threads: usize) -> Vec<T>
where
    T: BesselFloat + Send + Sync + 'static,
{
    let orders_chunks: Vec<Vec<i32>> = orders
        .chunks((orders.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let x_arc = Arc::new(x);
    let results = Arc::new(Mutex::new(vec![T::zero(); orders.len()]));

    let mut handles = vec![];

    for (thread_idx, chunk) in orders_chunks.into_iter().enumerate() {
        let x_ref = Arc::clone(&x_arc);
        let results_ref = Arc::clone(&results);
        let start_idx = thread_idx * chunk.len();

        handles.push(thread::spawn(move || {
            for (local_idx, &n) in chunk.iter().enumerate() {
                let global_idx = start_idx + local_idx;
                let result = bessj(n, *x_ref);
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bessj0_f32() {
        // Test cases from known values
        assert_abs_diff_eq!(bessj0(0.0_f32), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj0(1.0_f32), 0.7651977, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj0(2.0_f32), 0.22389078, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj0(5.0_f32), -0.17759677, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj0_f64() {
        assert_abs_diff_eq!(bessj0(0.0_f64), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj0(1.0_f64), 0.7651976865579666, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj0(2.0_f64), 0.22389077914123567, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj0(5.0_f64), -0.1775967713143383, epsilon = 1e-12);
    }

    #[test]
    fn test_bessj1_f32() {
        assert_abs_diff_eq!(bessj1(0.0_f32), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj1(1.0_f32), 0.44005059, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj1(2.0_f32), 0.5767248, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj1(5.0_f32), -0.3275791, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj1_f64() {
        assert_abs_diff_eq!(bessj1(0.0_f64), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj1(1.0_f64), 0.44005058574493355, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj1(2.0_f64), 0.5767248077568734, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj1(5.0_f64), -0.3275791375914652, epsilon = 1e-12);
    }

    #[test]
    fn test_bessj_f32() {
        // Test J₂(x) values
        assert_abs_diff_eq!(bessj(2, 0.0_f32), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj(2, 1.0_f32), 0.1149035, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj(2, 2.0_f32), 0.3528340, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj(2, 5.0_f32), 0.0465651, epsilon = 1e-6);
        
        // Test J₃(x) values
        assert_abs_diff_eq!(bessj(3, 1.0_f32), 0.0195634, epsilon = 1e-6);
        assert_abs_diff_eq!(bessj(3, 3.0_f32), 0.3090627, epsilon = 1e-6);
    }

    #[test]
    fn test_bessj_f64() {
        assert_abs_diff_eq!(bessj(2, 0.0_f64), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj(2, 1.0_f64), 0.11490348493190047, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj(2, 2.0_f64), 0.3528340286156377, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj(2, 5.0_f64), 0.046565116277752214, epsilon = 1e-12);
        
        assert_abs_diff_eq!(bessj(3, 1.0_f64), 0.019563353982668405, epsilon = 1e-12);
        assert_abs_diff_eq!(bessj(3, 3.0_f64), 0.30906272225525163, epsilon = 1e-12);
    }

    #[test]
    fn test_negative_x() {
        // Bessel functions with negative x
        assert_abs_diff_eq!(bessj(2, -1.0_f64), bessj(2, 1.0_f64), epsilon = 1e-12);
        assert_abs_diff_eq!(bessj(3, -2.0_f64), -bessj(3, 2.0_f64), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "Index n less than 2 in bessj")]
    fn test_invalid_order() {
        bessj(1, 1.0_f64);
    }

#[test]
fn test_multithreaded_bessj() {
    let orders = vec![2, 3, 4, 5, 6, 7, 8, 9];
    let x = 2.5_f64;
    
    // Single-threaded reference
    let single_threaded: Vec<f64> = orders.iter().map(|&n| bessj(n, x)).collect();
    
    // Multi-threaded
    let multi_threaded = bessj_multithreaded(&orders, x, 4);
    
    for (i, (&st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
        let diff = (st - *mt).abs();
        assert!(
            diff < 1e-12,
            "Mismatch at index {} (order {}): single_threaded={}, multi_threaded={}, diff={}",
            i, orders[i], st, mt, diff
        );
    }
}
    #[test]
    fn test_edge_cases() {
        // Zero input
        assert_eq!(bessj(2, 0.0_f64), 0.0);
        assert_eq!(bessj(3, 0.0_f64), 0.0);
        
        // Large orders
        assert_abs_diff_eq!(bessj(10, 20.0_f64), 0.186482, epsilon = 1e-6);
        
        // Large x values
        assert_abs_diff_eq!(bessj(5, 100.0_f64), 0.020_789, epsilon = 1e-6);
    }

    #[test]
    fn test_consistency_between_precisions() {
        // Test that f32 and f64 give consistent results (within precision limits)
        let test_cases = [(2, 1.0), (3, 2.0), (4, 5.0), (5, 10.0)];
        
        for &(n, x) in &test_cases {
            let f32_result = bessj(n, x as f32);
            let f64_result = bessj(n, x as f64);
            
            assert_abs_diff_eq!(f32_result as f64, f64_result, epsilon = 1e-6);
        }
    }
}
