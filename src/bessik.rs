use std::f64::consts::{PI, FRAC_PI_2};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const EPS: f64 = 1.0e-10;
const FPMIN: f64 = 1.0e-30;
const MAXIT: usize = 10000;
const XMIN: f64 = 2.0;

/// Modified Bessel functions I and K and their derivatives
/// Returns (I_ν, K_ν, I'_ν, K'_ν)
pub fn bessik(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
    if x <= 0.0 || xnu < 0.0 {
        return Err("x must be > 0 and xnu must be >= 0".to_string());
    }

    let nl = (xnu + 0.5) as i32;
    let xmu = xnu - nl as f64;
    let xmu2 = xmu * xmu;

    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;

    // Compute continued fraction for I_ν
    let (ril, ripl) = bessik_continued_fraction(x, xmu, xi, xi2)?;

    // Compute K_ν using different methods based on x
    let (rkmu, rk1) = if x < XMIN {
        bessik_small_x(x, xmu, xmu2)?
    } else {
        bessik_large_x(x, xmu, xmu2, xi)?
    };

    // Compute derivatives and apply recurrence
    let (ri, rip, rk, rkp) = bessik_finalize(x, xnu, xmu, nl, xi, ril, ripl, rkmu, rk1);

    Ok((ri, rk, rip, rkp))
}

/// Continued fraction evaluation for I_ν
fn bessik_continued_fraction(x: f64, xmu: f64, xi: f64, xi2: f64) -> Result<(f64, f64), String> {
    let mut h = xmu * xi;
    if h < FPMIN {
        h = FPMIN;
    }

    let mut b = xi2 * xmu;
    let mut d = 0.0;
    let mut c = h;

    for i in 1..=MAXIT {
        b += xi2;
        d = 1.0 / (b + d);
        c = b + 1.0 / c;
        let del = c * d;
        h *= del;

        if (del - 1.0).abs() < EPS {
            let mut ril = FPMIN;
            let mut ripl = h * ril;

            return Ok((ril, ripl));
        }
    }

    Err("Continued fraction failed to converge in bessik".to_string())
}

/// Compute K_ν for small x (x < XMIN)
fn bessik_small_x(x: f64, xmu: f64, xmu2: f64) -> Result<(f64, f64), String> {
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

    let (gam1, gam2, gampl, gammi) = beschb(xmu)?;

    let mut ff = fact * (gam1 * e.cosh() + gam2 * fact2 * d);
    let mut sum = ff;

    let e_val = e.exp();
    let mut p = 0.5 * e_val / gampl;
    let mut q = 0.5 / (e_val * gammi);

    let mut c = 1.0;
    let d_val = x2 * x2;
    let mut sum1 = p;

    for i in 1..=MAXIT {
        ff = (i as f64 * ff + p + q) / (i as f64 * i as f64 - xmu2);
        c *= d_val / i as f64;
        p /= i as f64 - xmu;
        q /= i as f64 + xmu;
        let del = c * ff;
        sum += del;
        let del1 = c * (p - i as f64 * ff);
        sum1 += del1;

        if del.abs() < sum.abs() * EPS {
            let rkmu = sum;
            let rk1 = sum1 * (2.0 / x);
            return Ok((rkmu, rk1));
        }
    }

    Err("Series failed to converge for small x in bessik".to_string())
}

/// Compute K_ν for large x (x >= XMIN)
fn bessik_large_x(x: f64, xmu: f64, xmu2: f64, xi: f64) -> Result<(f64, f64), String> {
    let b = 2.0 * (1.0 + x);
    let mut d = 1.0 / b;
    let mut h = d;
    let mut delh = d;

    let mut q1 = 0.0;
    let mut q2 = 1.0;
    let a1 = 0.25 - xmu2;
    let mut q = a1;
    let mut c = a1;
    let mut a = -a1;
    let mut s = 1.0 + q * delh;

    for i in 2..=MAXIT {
        a -= 2.0 * (i - 1) as f64;
        c = -a * c / i as f64;
        let qnew = (q1 - b * q2) / a;
        q1 = q2;
        q2 = qnew;
        q += c * qnew;
        let b_val = b + 2.0;
        d = 1.0 / (b_val + a * d);
        delh = (b_val * d - 1.0) * delh;
        h += delh;
        let dels = q * delh;
        s += dels;

        if dels.abs() < s.abs() * EPS {
            let h_val = a1 * h;
            let rkmu = (PI / (2.0 * x)).sqrt() * (-x).exp() / s;
            let rk1 = rkmu * (xmu + x + 0.5 - h_val) * xi;
            return Ok((rkmu, rk1));
        }
    }

    Err("Continued fraction failed to converge for large x in bessik".to_string())
}

/// Final computation and recurrence application
fn bessik_finalize(
    x: f64,
    xnu: f64,
    xmu: f64,
    nl: i32,
    xi: f64,
    mut ril: f64,
    mut ripl: f64,
    mut rkmu: f64,
    mut rk1: f64,
) -> (f64, f64, f64, f64) {
    // Apply recurrence for I_ν
    let mut fact = xnu * xi;
    for l in (1..=nl).rev() {
        let ritemp = fact * ril + ripl;
        fact -= xi;
        ripl = fact * ritemp + ril;
        ril = ritemp;
    }

    let f = ripl / ril;

    // Compute derivatives
    let rkmup = xmu * xi * rkmu - rk1;
    let rimu = xi / (f * rkmu - rkmup);

    let ri = rimu * ril;
    let rip = rimu * ripl;

    // Apply recurrence for K_ν
    for i in 1..=nl {
        let rktemp = (xmu + i as f64) * (2.0 / x) * rk1 + rkmu;
        rkmu = rk1;
        rk1 = rktemp;
    }

    let rkp = xnu * xi * rkmu - rk1;

    (ri, rkmu, rip, rkp)
}

/// Chebyshev expansion for Γ functions
fn beschb(x: f64) -> Result<(f64, f64, f64, f64), String> {
    const NUSE1: usize = 7;
    const NUSE2: usize = 8;

    let coef1 = [
        -1.142022680371168e0, 6.5165112670737e-3,
        3.087090173086e-4, -3.4706269649e-6,
        6.9437664e-9, 3.67795e-11, -1.356e-13
    ];

    let coef2 = [
        1.843740587300905e0, -7.68528408447867e-2,
        1.2719271366546e-3, -4.9717367042e-6,
        -3.31261198e-8, 2.423096e-10, -1.702e-13,
        -1.49e-15
    ];

    let xx = 8.0 * x * x - 1.0;
    let mut gam1 = chebyshev_eval(xx, &coef1);
    let mut gam2 = chebyshev_eval(xx, &coef2);
    let gampl = gam2 - x * gam1;
    let gammi = gam2 + x * gam1;

    Ok((gam1, gam2, gampl, gammi))
}

/// Chebyshev polynomial evaluation
fn chebyshev_eval(x: f64, coef: &[f64]) -> f64 {
    let mut d = 0.0;
    let mut dd = 0.0;
    let y = 2.0 * x;

    for &c in coef.iter().rev() {
        let temp = d;
        d = y * d - dd + c;
        dd = temp;
    }

    0.5 * (d - dd)
}

/// Thread-safe cache for Bessel I and K values
#[derive(Clone)]
pub struct BesselIKCache {
    cache: Arc<Mutex<std::collections::HashMap<(i32, i32), (f64, f64, f64, f64)>>>,
}

impl BesselIKCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get bessik values with caching
    pub fn get(&self, x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        if x <= 0.0 || xnu < 0.0 {
            return Err("Invalid arguments".to_string());
        }

        let key = ((x * 100.0) as i32, (xnu * 100.0) as i32);
        let mut cache = self.cache.lock().unwrap();

        if let Some(&result) = cache.get(&key) {
            return Ok(result);
        }

        let result = bessik(x, xnu)?;
        cache.insert(key, result);
        Ok(result)
    }

    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

/// Parallel batch computation for multiple x values
pub fn bessik_batch_x_parallel(x_values: &[f64], xnu: f64) -> Vec<Result<(f64, f64, f64, f64), String>> {
    x_values.par_iter()
        .map(|&x| bessik(x, xnu))
        .collect()
}

/// Parallel batch computation for multiple orders
pub fn bessik_batch_nu_parallel(x: f64, xnu_values: &[f64]) -> Vec<Result<(f64, f64, f64, f64), String>> {
    xnu_values.par_iter()
        .map(|&xnu| bessik(x, xnu))
        .collect()
}

/// Compute only I_ν(x)
pub fn bessi(x: f64, xnu: f64) -> Result<f64, String> {
    bessik(x, xnu).map(|(i, _, _, _)| i)
}

/// Compute only K_ν(x)
pub fn bessk(x: f64, xnu: f64) -> Result<f64, String> {
    bessik(x, xnu).map(|(_, k, _, _)| k)
}

/// Compute only I'_ν(x)
pub fn bessip(x: f64, xnu: f64) -> Result<f64, String> {
    bessik(x, xnu).map(|(_, _, ip, _)| ip)
}

/// Compute only K'_ν(x)
pub fn besskp(x: f64, xnu: f64) -> Result<f64, String> {
    bessik(x, xnu).map(|(_, _, _, kp)| kp)
}

/// Compute the ratio I_ν(x)/I_ν₋₁(x)
pub fn bessi_ratio(x: f64, xnu: f64) -> Result<f64, String> {
    if xnu < 1.0 {
        return Err("xnu must be >= 1 for ratio computation".to_string());
    }
    let i_nu = bessi(x, xnu)?;
    let i_nu_minus_1 = bessi(x, xnu - 1.0)?;
    Ok(i_nu / i_nu_minus_1)
}

/// Compute the ratio K_ν(x)/K_ν₋₁(x)
pub fn bessk_ratio(x: f64, xnu: f64) -> Result<f64, String> {
    if xnu < 1.0 {
        return Err("xnu must be >= 1 for ratio computation".to_string());
    }
    let k_nu = bessk(x, xnu)?;
    let k_nu_minus_1 = bessk(x, xnu - 1.0)?;
    Ok(k_nu / k_nu_minus_1)
}

/// Asymptotic expansion for large x
pub fn bessik_asymptotic(x: f64, xnu: f64) -> Result<(f64, f64), String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }

    let z = x;
    let mu = 4.0 * xnu * xnu;
    let mut term = 1.0;
    let mut sum_i = 1.0;
    let mut sum_k = 1.0;

    for k in 1..=10 {
        term *= (mu - (2.0 * k as f64 - 1.0).powi(2)) / (8.0 * z * k as f64);
        sum_i += term;
        if k % 2 == 0 {
            sum_k += term;
        } else {
            sum_k -= term;
        }
    }

    let prefactor_i = (z.exp() / (2.0 * PI * z).sqrt()) * sum_i;
    let prefactor_k = (PI / (2.0 * z)).sqrt() * (-z).exp() * sum_k;

    Ok((prefactor_i, prefactor_k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessik_basic() {
        // Test known values
        let (i0, k0, _, _) = bessik(1.0, 0.0).unwrap();
        assert_relative_eq!(i0, 1.266065878, epsilon = 1e-6);
        assert_relative_eq!(k0, 0.421024438, epsilon = 1e-6);

        let (i1, k1, _, _) = bessik(1.0, 1.0).unwrap();
        assert_relative_eq!(i1, 0.565159104, epsilon = 1e-6);
        assert_relative_eq!(k1, 0.601907230, epsilon = 1e-6);
    }

    #[test]
    fn test_bessik_error_cases() {
        assert!(bessik(0.0, 1.0).is_err());
        assert!(bessik(1.0, -1.0).is_err());
    }

    #[test]
    fn test_bessik_small_x() {
        // Test small x regime
        let result = bessik(1.5, 0.5).unwrap();
        assert!(result.0.is_finite());
        assert!(result.1.is_finite());
        assert!(result.2.is_finite());
        assert!(result.3.is_finite());
    }

    #[test]
    fn test_bessik_large_x() {
        // Test large x regime
        let result = bessik(10.0, 2.0).unwrap();
        assert!(result.0.is_finite());
        assert!(result.1.is_finite());
        assert!(result.2.is_finite());
        assert!(result.3.is_finite());
    }

    #[test]
    fn test_beschb() {
        let (gam1, gam2, gampl, gammi) = beschb(0.5).unwrap();
        assert!(gam1.is_finite());
        assert!(gam2.is_finite());
        assert!(gampl.is_finite());
        assert!(gammi.is_finite());
    }

    #[test]
    fn test_individual_functions() {
        let i0 = bessi(1.0, 0.0).unwrap();
        let k0 = bessk(1.0, 0.0).unwrap();
        assert_relative_eq!(i0, 1.266065878, epsilon = 1e-6);
        assert_relative_eq!(k0, 0.421024438, epsilon = 1e-6);
    }

    #[test]
    fn test_ratios() {
        let ratio_i = bessi_ratio(2.0, 1.0).unwrap();
        let i1 = bessi(2.0, 1.0).unwrap();
        let i0 = bessi(2.0, 0.0).unwrap();
        assert_relative_eq!(ratio_i, i1 / i0, epsilon = 1e-6);

        let ratio_k = bessk_ratio(2.0, 1.0).unwrap();
        let k1 = bessk(2.0, 1.0).unwrap();
        let k0 = bessk(2.0, 0.0).unwrap();
        assert_relative_eq!(ratio_k, k1 / k0, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [0.5, 1.0, 2.0, 5.0];
        let results = bessik_batch_x_parallel(&x_values, 0.5);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = bessik(x_values[i], 0.5).unwrap();
            assert_relative_eq!(result.as_ref().unwrap().0, expected.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_asymptotic_expansion() {
        let (i_asym, k_asym) = bessik_asymptotic(10.0, 2.0).unwrap();
        let (i_exact, k_exact, _, _) = bessik(10.0, 2.0).unwrap();
        
        assert_relative_eq!(i_asym, i_exact, epsilon = 1e-3);
        assert_relative_eq!(k_asym, k_exact, epsilon = 1e-3);
    }

    #[test]
    fn test_cache_functionality() {
        let cache = BesselIKCache::new();
        
        let result1 = cache.get(1.0, 0.5).unwrap();
        let result2 = cache.get(1.0, 0.5).unwrap(); // Should be cached
        
        assert_relative_eq!(result1.0, result2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.1, result2.1, epsilon = 1e-10);
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_bessik_small_x(c: &mut Criterion) {
        c.bench_function("bessik(1.0, 0.5)", |b| {
            b.iter(|| bessik(black_box(1.0), black_box(0.5)).unwrap())
        });
    }

    fn bench_bessik_large_x(c: &mut Criterion) {
        c.bench_function("bessik(10.0, 2.0)", |b| {
            b.iter(|| bessik(black_box(10.0), black_box(2.0)).unwrap())
        });
    }

    fn bench_bessik_batch(c: &mut Criterion) {
        let x_values: Vec<f64> = (1..=20).map(|x| x as f64 / 2.0).collect();
        
        c.bench_function("bessik_batch_x_parallel(20)", |b| {
            b.iter(|| bessik_batch_x_parallel(black_box(&x_values), black_box(1.0)))
        });
    }

    fn bench_bessi_individual(c: &mut Criterion) {
        c.bench_function("bessi(2.0, 1.0)", |b| {
            b.iter(|| bessi(black_box(2.0), black_box(1.0)).unwrap())
        });
    }

    fn bench_bessk_individual(c: &mut Criterion) {
        c.bench_function("bessk(2.0, 1.0)", |b| {
            b.iter(|| bessk(black_box(2.0), black_box(1.0)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_bessik_small_x, 
        bench_bessik_large_x,
        bench_bessik_batch,
        bench_bessi_individual,
        bench_bessk_individual
    );
    criterion_main!(benches);
}
