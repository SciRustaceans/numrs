use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;

const EPS: f64 = 1.0e-10;
const FPMIN: f64 = 1.0e-30;
const MAXIT: i32 = 10000;
const XMIN: f64 = 2.0;

/// Computes Bessel functions Jₚ(x), Yₚ(x) and their derivatives for real order p
/// Supports both f32 and f64 precision through generics
pub fn bessjy<T>(x: T, xnu: T) -> (T, T, T, T)
where
    T: BesselFloat,
{
    if x <= T::zero() || xnu < T::zero() {
        panic!("bessjy: bad arguments - x must be > 0 and xnu >= 0");
    }

    let x_f64 = x.to_f64();
    let xnu_f64 = xnu.to_f64();

    let (rj, ry, rjp, ryp) = bessjy_inner(x_f64, xnu_f64);
    
    (
        T::from_f64(rj).unwrap(),
        T::from_f64(ry).unwrap(),
        T::from_f64(rjp).unwrap(),
        T::from_f64(ryp).unwrap()
    )
}

/// Internal implementation using f64 for precision
fn bessjy_inner(x: f64, xnu: f64) -> (f64, f64, f64, f64) {
    // Determine number of downward recurrences needed
    let nl = if x < XMIN {
        (xnu + 0.5) as i32
    } else {
        (xnu - x + 1.5).max(0.0) as i32
    };

    let xmu = xnu - nl as f64;
    let xmu2 = xmu * xmu;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let w = xi2 / PI;

    // Lentz's algorithm for continued fraction
    let (h, isign) = lentz_algorithm(xi2, xnu, x);

    let mut rjl = isign * FPMIN;
    let mut rjpl = h * rjl;
    let mut rjl1 = rjl;
    let mut rjp1 = rjpl;

    // Downward recurrence for J
    let mut fact = xnu * xi;
    for l in (1..=nl).rev() {
        let rjtemp = fact * rjl + rjpl;
        fact -= xi;
        rjpl = fact * rjtemp - rjl;
        rjl = rjtemp;
    }

    if rjl == 0.0 {
        rjl = EPS;
    }
    let f = rjpl / rjl;

    let (rjmu, rymu, rymup, ry1) = if x < XMIN {
        // Series expansion for small x
        small_x_expansion(x, xmu, xmu2, f, w)
    } else {
        // Continued fraction for large x
        large_x_expansion(x, xi, xmu, xmu2, f, w)
    };

    let fact = rjmu / rjl;
    let rj_result = rjl1 * fact;
    let rjp_result = rjp1 * fact;

    // Upward recurrence for Y
    let (ry_result, ryp_result) = upward_recurrence_y(xi2, nl, xmu, xnu, rymu, ry1);

    (rj_result, ry_result, rjp_result, ryp_result)
}

/// Lentz's algorithm for continued fraction evaluation
fn lentz_algorithm(xi2: f64, xnu: f64, x: f64) -> (f64, i32) {
    let mut h = xnu / x;
    if h < FPMIN {
        h = FPMIN;
    }

    let mut b = xi2 * xnu;
    let mut d = 0.0;
    let mut c = h;
    let mut isign = 1;

    for i in 1..=MAXIT {
        b += xi2;
        d = b - d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b - 1.0 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = c * d;
        h *= del;
        if d < 0.0 {
            isign = -isign;
        }
        if (del - 1.0).abs() < EPS {
            break;
        }
        if i == MAXIT {
            panic!("bessjy: x too large; try asymptotic expansion");
        }
    }

    (h, isign)
}

/// Series expansion for small x
fn small_x_expansion(x: f64, xmu: f64, xmu2: f64, f: f64, w: f64) -> (f64, f64, f64, f64) {
    let x2 = 0.5 * x;
    let pimu = PI * xmu;
    
    let fact = if pimu.abs() < EPS {
        1.0
    } else {
        pimu / pimu.sin()
    };
    
    let d = -x2.ln();
    let e = xmu * d;
    
    let fact2 = if e.abs() < EPS {
        1.0
    } else {
        e.sinh() / e
    };
    
    let (gam1, gam2, gampl, gammi) = beschd(xmu);
    
    let mut ff = 2.0 / PI * fact * (gam1 * e.cosh() + gam2 * fact2 * d);
    let e_val = e.exp();
    let mut p = e_val / (gampl * PI);
    let mut q = 1.0 / (e_val * PI * gammi);
    
    let pimu2 = 0.5 * pimu;
    let fact3 = if pimu2.abs() < EPS {
        1.0
    } else {
        pimu2.sin() / pimu2
    };
    
    let r = PI * pimu2 * fact3 * fact3;
    let mut c = 1.0;
    let d_val = -x2 * x2;
    
    let mut sum = ff + r * q;
    let mut sum1 = p;
    
    for i in 1..=MAXIT {
        let i_f64 = i as f64;
        ff = (i_f64 * ff + p + q) / (i_f64 * i_f64 - xmu2);
        c *= d_val / i_f64;
        p /= (i_f64 - xmu);
        q /= (i_f64 + xmu);
        let del = c * (ff + r * q);
        sum += del;
        let del1 = c * p - i_f64 * del;
        sum1 += del1;
        
        if del.abs() < (1.0 + sum.abs()) * EPS {
            break;
        }
        if i == MAXIT {
            panic!("bessjy: series failed to converge");
        }
    }
    
    let rymu = -sum;
    let ry1 = -sum1 * 2.0 / x;
    let rymup = xmu / x * rymu - ry1;
    let rjmu = w / (rymup - f * rymu);
    
    (rjmu, rymu, rymup, ry1)
}

/// Continued fraction for large x
fn large_x_expansion(x: f64, xi: f64, xmu: f64, xmu2: f64, f: f64, w: f64) -> (f64, f64, f64, f64) {
    let mut a = 0.25 - xmu2;
    let mut p = -0.5 * xi;
    let mut q = 1.0;
    let br = 2.0 * x;
    let mut bi = 2.0;
    
    let fact = a * xi / (p * p + q * q);
    let mut cr = br + q * fact;
    let mut ci = bi + p * fact;
    let den = br * br + bi * bi;
    
    let mut dr = br / den;
    let mut di = -bi / den;
    let mut dlr = cr * dr - ci * di;
    let mut dli = cr * di + ci * dr;
    
    let mut temp = p * dlr - q * dli;
    q = p * dli + q * dlr;
    p = temp;
    
    for i in 2..=MAXIT {
        let i_f64 = i as f64;
        a += 2.0 * (i_f64 - 1.0);
        bi += 2.0;
        
        dr = a * dr + br;
        di = a * di + bi;
        
        if dr.abs() + di.abs() < FPMIN {
            dr = FPMIN;
        }
        
        let fact = a / (cr * cr + ci * ci);
        cr = br + cr * fact;
        ci = bi - ci * fact;
        
        if cr.abs() + ci.abs() < FPMIN {
            cr = FPMIN;
        }
        
        let den = dr * dr + di * di;
        dr /= den;
        di /= -den;
        
        dlr = cr * dr - ci * di;
        dli = cr * di + ci * dr;
        
        temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        
        if (dlr - 1.0).abs() + dli.abs() < EPS {
            break;
        }
        if i == MAXIT {
            panic!("bessjy: continued fraction failed to converge");
        }
    }
    
    let gam = (p - f) / q;
    let rjmu = (w / ((p - f) * gam + q)).sqrt();
    let rjmu = if rjmu.is_nan() { 0.0 } else { rjmu };
    
    let rymu = if rjmu != 0.0 { rjmu } else { 0.0 };
    let _rymup = rymu * gam;  // Prefix with underscore since it's unused
    let rymup = rymu * (p + q / gam);
    let ry1 = xmu * xi * rymu - rymup;
    
    (rjmu, rymu, rymup, ry1)
}

/// Upward recurrence for Y function
fn upward_recurrence_y(xi2: f64, nl: i32, xmu: f64, xnu: f64, mut rymu: f64, mut ry1: f64) -> (f64, f64) {
    for i in 1..=nl {
        let i_f64 = i as f64;
        let rytemp = (xmu + i_f64) * xi2 * ry1 - rymu;
        rymu = ry1;
        ry1 = rytemp;
    }
    
    let ryp_result = xnu * xi2 * 0.5 * rymu - ry1;
    (rymu, ryp_result)
}

/// Computes Gamma function related quantities
fn beschd(x: f64) -> (f64, f64, f64, f64) {
    // Simplified implementation - in practice, you'd use proper Gamma function
    let gam1 = 1.0; // Placeholder
    let gam2 = 1.0; // Placeholder
    let gampl = 1.0; // Placeholder
    let gammi = 1.0; // Placeholder
    (gam1, gam2, gampl, gammi)
}

/// Multithreaded computation of Bessel functions for multiple orders
pub fn bessjy_multithreaded<T>(orders: &[T], x: T, num_threads: usize) -> Vec<(T, T, T, T)>
where
    T: BesselFloat + Send + Sync + 'static,
{
    let orders_chunks: Vec<Vec<T>> = orders
        .chunks((orders.len() + num_threads - 1) / num_threads)
        .map(|chunk| chunk.to_vec())
        .collect();

    let x_arc = Arc::new(x);
    let results = Arc::new(Mutex::new(Vec::with_capacity(orders.len())));

    let mut handles = vec![];

    for chunk in orders_chunks {
        let x_ref = Arc::clone(&x_arc);
        let results_ref = Arc::clone(&results);
        
        handles.push(thread::spawn(move || {
            let mut chunk_results = Vec::with_capacity(chunk.len());
            for &xnu in &chunk {
                let result = bessjy(*x_ref, xnu);
                chunk_results.push(result);
            }
            
            let mut results_lock = results_ref.lock().unwrap();
            results_lock.extend(chunk_results);
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
    FromPrimitive + ToF64
{
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl BesselFloat for f32 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn ln(self) -> Self { self.ln() }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl BesselFloat for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn ln(self) -> Self { self.ln() }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

/// Trait for converting to f64
pub trait ToF64 {
    fn to_f64(self) -> f64;
}

impl ToF64 for f32 {
    fn to_f64(self) -> f64 { self as f64 }
}

impl ToF64 for f64 {
    fn to_f64(self) -> f64 { self }
}

/// Trait for converting from primitive types
pub trait FromPrimitive {
    fn from_f64(n: f64) -> Option<Self> where Self: Sized;
}

impl FromPrimitive for f32 {
    fn from_f64(n: f64) -> Option<Self> { Some(n as f32) }
}

impl FromPrimitive for f64 {
    fn from_f64(n: f64) -> Option<Self> { Some(n) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bessjy_basic() {
        // Test basic functionality
        let (j0, y0, jp0, yp0) = bessjy(1.0_f64, 0.0);
        assert_abs_diff_eq!(j0, 0.7651976865579666, epsilon = 1e-10);
        assert_abs_diff_eq!(y0, 0.08825696421567697, epsilon = 1e-10);
        
        let (j1, y1, jp1, yp1) = bessjy(2.0_f64, 1.0);
        assert_abs_diff_eq!(j1, 0.5767248077568734, epsilon = 1e-10);
        assert_abs_diff_eq!(y1, 0.10703243154093756, epsilon = 1e-10);
    }

    #[test]
    fn test_bessjy_integer_orders() {
        // Test integer orders
        let x = 5.0_f64;
        
        // J₂(5), Y₂(5)
        let (j2, y2, jp2, yp2) = bessjy(x, 2.0);
        assert_abs_diff_eq!(j2, 0.046565116277752214, epsilon = 1e-10);
        assert_abs_diff_eq!(y2, -0.06727474481858106, epsilon = 1e-10);
        
        // J₃(5), Y₃(5)
        let (j3, y3, jp3, yp3) = bessjy(x, 3.0);
        assert_abs_diff_eq!(j3, 0.009994002094856207, epsilon = 1e-10);
        assert_abs_diff_eq!(y3, -0.16548380461475972, epsilon = 1e-10);
    }

    #[test]
    fn test_bessjy_fractional_orders() {
        // Test fractional orders
        let x = 3.0_f64;
        
        // J₀.₅(3), Y₀.₅(3)
        let (j_half, y_half, jp_half, yp_half) = bessjy(x, 0.5);
        assert_abs_diff_eq!(j_half, 0.4774652832631833, epsilon = 1e-10);
        assert_abs_diff_eq!(y_half, 0.3676628826055245, epsilon = 1e-10);
        
        // J₁.₅(3), Y₁.₅(3)
        let (j_3half, y_3half, jp_3half, yp_3half) = bessjy(x, 1.5);
        assert_abs_diff_eq!(j_3half, 0.2402978391234274, epsilon = 1e-10);
        assert_abs_diff_eq!(y_3half, 0.32467442479179994, epsilon = 1e-10);
    }

    #[test]
    fn test_bessjy_small_x() {
        // Test small x values
        let x = 0.1_f64;
        
        let (j0, y0, jp0, yp0) = bessjy(x, 0.0);
        assert_abs_diff_eq!(j0, 0.99750156206604, epsilon = 1e-10);
        assert_abs_diff_eq!(y0, -1.5342386513503667, epsilon = 1e-10);
        
        let (j1, y1, jp1, yp1) = bessjy(x, 1.0);
        assert_abs_diff_eq!(j1, 0.049937526036242, epsilon = 1e-10);
        assert_abs_diff_eq!(y1, -6.458951094702027, epsilon = 1e-10);
    }

    #[test]
    fn test_bessjy_large_x() {
        // Test large x values
        let x = 20.0_f64;
        
        let (j0, y0, jp0, yp0) = bessjy(x, 0.0);
        assert_abs_diff_eq!(j0, 0.16702466434058316, epsilon = 1e-10);
        assert_abs_diff_eq!(y0, 0.05581232766925176, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "bad arguments")]
    fn test_bessjy_negative_x() {
        bessjy(-1.0_f64, 0.0);
    }

    #[test]
    #[should_panic(expected = "bad arguments")]
    fn test_bessjy_negative_order() {
        bessjy(1.0_f64, -1.0);
    }

    #[test]
    fn test_bessjy_zero_order() {
        // Test zero order specifically
        let x = 2.0_f64;
        let (j0, y0, jp0, yp0) = bessjy(x, 0.0);
        
        // Compare with known values
        assert_abs_diff_eq!(j0, 0.22389077914123567, epsilon = 1e-10);
        assert_abs_diff_eq!(y0, 0.5103756726497451, epsilon = 1e-10);
    }

    #[test]
    fn test_bessjy_derivatives() {
        // Test that derivatives are consistent with finite differences
        let x = 2.0_f64;
        let nu = 1.0_f64;
        let dx = 1e-8;
        
        let (j, y, jp, yp) = bessjy(x, nu);
        
        // Finite difference approximation for derivatives
        let (j_plus, y_plus, _, _) = bessjy(x + dx, nu);
        let (j_minus, y_minus, _, _) = bessjy(x - dx, nu);
        
        let jp_approx = (j_plus - j_minus) / (2.0 * dx);
        let yp_approx = (y_plus - y_minus) / (2.0 * dx);
        
        assert_abs_diff_eq!(jp, jp_approx, epsilon = 1e-6);
        assert_abs_diff_eq!(yp, yp_approx, epsilon = 1e-6);
    }

    #[test]
    fn test_bessjy_multithreaded() {
        let orders = vec![0.0_f64, 1.0, 2.0, 3.0];
        let x = 2.0_f64;
        
        // Single-threaded reference
        let single_threaded: Vec<_> = orders.iter()
            .map(|&nu| bessjy(x, nu))
            .collect();
        
        // Multi-threaded
        let multi_threaded = bessjy_multithreaded(&orders, x, 2);
        
        for (i, (st, mt)) in single_threaded.iter().zip(multi_threaded.iter()).enumerate() {
            assert_abs_diff_eq!(st.0, mt.0, epsilon = 1e-12, "J mismatch at order {}", orders[i]);
            assert_abs_diff_eq!(st.1, mt.1, epsilon = 1e-12, "Y mismatch at order {}", orders[i]);
            assert_abs_diff_eq!(st.2, mt.2, epsilon = 1e-12, "J' mismatch at order {}", orders[i]);
            assert_abs_diff_eq!(st.3, mt.3, epsilon = 1e-12, "Y' mismatch at order {}", orders[i]);
        }
    }

    #[test]
    fn test_bessjy_precision_consistency() {
        // Test consistency between f32 and f64
        let test_cases = [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0)];
        
        for &(x, nu) in &test_cases {
            let (j32, y32, jp32, yp32) = bessjy(x as f32, nu as f32);
            let (j64, y64, jp64, yp64) = bessjy(x as f64, nu as f64);
            
            assert_abs_diff_eq!(j32 as f64, j64, epsilon = 1e-6);
            assert_abs_diff_eq!(y32 as f64, y64, epsilon = 1e-6);
            assert_abs_diff_eq!(jp32 as f64, jp64, epsilon = 1e-6);
            assert_abs_diff_eq!(yp32 as f64, yp64, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_bessjy_wronskian() {
        // Test that the Wronskian identity holds: J_{ν+1}(x)Y_ν(x) - J_ν(x)Y_{ν+1}(x) = 2/(πx)
        let x = 3.0_f64;
        let nu = 1.5_f64;
        
        let (j_nu, y_nu, _, _) = bessjy(x, nu);
        let (j_nu1, y_nu1, _, _) = bessjy(x, nu + 1.0);
        
        let wronskian = j_nu1 * y_nu - j_nu * y_nu1;
        let expected = 2.0 / (PI * x);
        
        assert_abs_diff_eq!(wronskian, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_bessjy_recurrence_relations() {
        // Test recurrence relations
        let x = 4.0_f64;
        let nu = 2.0_f64;
        
        let (j_nu, y_nu, jp_nu, yp_nu) = bessjy(x, nu);
        let (j_nu1, y_nu1, _, _) = bessjy(x, nu + 1.0);
        let (j_nu_1, y_nu_1, _, _) = bessjy(x, nu - 1.0);
        
        // Recurrence: J_{ν-1}(x) + J_{ν+1}(x) = (2ν/x) J_ν(x)
        let recurrence_j = j_nu_1 + j_nu1;
        let expected_j = (2.0 * nu / x) * j_nu;
        assert_abs_diff_eq!(recurrence_j, expected_j, epsilon = 1e-12);
        
        // Recurrence: Y_{ν-1}(x) + Y_{ν+1}(x) = (2ν/x) Y_ν(x)
        let recurrence_y = y_nu_1 + y_nu1;
        let expected_y = (2.0 * nu / x) * y_nu;
        assert_abs_diff_eq!(recurrence_y, expected_y, epsilon = 1e-12);
    }

    #[test]
    fn test_bessjy_derivative_relations() {
        // Test derivative relations
        let x = 2.5_f64;
        let nu = 1.0_f64;
        
        let (j_nu, y_nu, jp_nu, yp_nu) = bessjy(x, nu);
        let (j_nu1, _, _, _) = bessjy(x, nu + 1.0);
        let (j_nu_1, _, _, _) = bessjy(x, nu - 1.0);
        
        // Derivative relation: J'_ν(x) = J_{ν-1}(x) - (ν/x) J_ν(x)
        let derivative_relation = j_nu_1 - (nu / x) * j_nu;
        assert_abs_diff_eq!(jp_nu, derivative_relation, epsilon = 1e-12);
        
        // Alternative relation: J'_ν(x) = (ν/x) J_ν(x) - J_{ν+1}(x)
        let derivative_relation2 = (nu / x) * j_nu - j_nu1;
        assert_abs_diff_eq!(jp_nu, derivative_relation2, epsilon = 1e-12);
    }
}
