use std::f64::consts::{PI, FRAC_PI_2};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const RTPIO2: f64 = FRAC_PI_2.sqrt(); // √(π/2) ≈ 1.253314137

/// Gamma function using Lanczos approximation
fn gamma(x: f64) -> f64 {
    // Lanczos approximation for gamma function
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
    
    let mut x = x - 1.0;
    let mut t = COEFFS[0];
    
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        t += coeff / (x + i as f64);
    }
    
    let sqrt_2pi = (2.0 * PI).sqrt();
    (sqrt_2pi * t).ln().exp() * (x + 0.5).powf(x + 0.5) * (-x - 0.5).exp()
}

/// Series expansion for Bessel J function
fn bessj_series(x: f64, xnu: f64) -> (f64, f64) {
    // Series expansion for J_ν(x)
    let mut sum = 0.0;
    let mut sum_deriv = 0.0;
    let x2 = x * x / 4.0;
    let mut term = 1.0;
    let gamma_nu = gamma(xnu + 1.0);
    
    for k in 0..20 {
        term *= -x2 / ((k as f64 + 1.0) * (k as f64 + xnu + 1.0));
        sum += term;
        sum_deriv += (2.0 * k as f64 + xnu + 1.0) * term / x;
    }
    
    let j = (x / 2.0).powf(xnu) * sum / gamma_nu;
    let jp = (x / 2.0).powf(xnu - 1.0) * sum_deriv / (2.0 * gamma_nu);
    
    (j, jp)
}

/// Asymptotic expansion for Bessel J function for large x
fn bessj_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
    // Asymptotic expansion for J_ν(x)
    let z = x;
    let mu = 4.0 * xnu * xnu;
    let mut p = 1.0;
    let mut q = 0.0;
    let mut term = 1.0;
    
    for k in 1..10 {
        term *= (mu - (2.0 * k as f64 - 1.0).powi(2)) / (8.0 * z * k as f64);
        if k % 2 == 0 {
            p += term;
        } else {
            q += term;
        }
    }
    
    let phase = z - xnu * PI / 2.0 - PI / 4.0;
    let (sin_phase, cos_phase) = phase.sin_cos();
    
    let j = (2.0 / (PI * z)).sqrt() * (p * cos_phase - q * sin_phase);
    let jp = -(2.0 / (PI * z)).sqrt() * (p * sin_phase + q * cos_phase);
    
    (j, jp)
}

/// Asymptotic expansion for Bessel Y function for large x
fn bessy_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
    // Asymptotic expansion for Y_ν(x)
    let z = x;
    let mu = 4.0 * xnu * xnu;
    let mut p = 1.0;
    let mut q = 0.0;
    let mut term = 1.0;
    
    for k in 1..10 {
        term *= (mu - (2.0 * k as f64 - 1.0).powi(2)) / (8.0 * z * k as f64);
        if k % 2 == 0 {
            p += term;
        } else {
            q += term;
        }
    }
    
    let phase = z - xnu * PI / 2.0 - PI / 4.0;
    let (sin_phase, cos_phase) = phase.sin_cos();
    
    let y = (2.0 / (PI * z)).sqrt() * (p * sin_phase + q * cos_phase);
    let yp = (2.0 / (PI * z)).sqrt() * (p * cos_phase - q * sin_phase);
    
    (y, yp)
}

/// Bessel J and Y functions with derivatives
fn bessjy(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
    if x <= 0.0 || xnu < 0.0 {
        return Err("Invalid arguments in bessjy".to_string());
    }
    
    if x < 2.0 {
        // For small x, use series expansion
        let (j, jp) = bessj_series(x, xnu);
        
        // For Y_ν, we need to handle the case carefully near x=0
        if x < 1e-8 {
            if xnu == 0.0 {
                let y = 2.0 / PI * (x.ln() + 0.57721566490153286060); // Euler-Mascheroni constant
                let yp = 2.0 / (PI * x);
                Ok((j, y, jp, yp))
            } else {
                let gamma_nu = gamma(xnu);
                let y = -gamma_nu / PI * (2.0 / x).powf(xnu);
                let yp = xnu * gamma_nu / PI * (2.0 / x).powf(xnu + 1.0);
                Ok((j, y, jp, yp))
            }
        } else {
            // Use relation between J_ν and J_{-ν} for Y_ν
            let (j_minus, jp_minus) = bessj_series(x, -xnu);
            let y = (j * (xnu * PI).cos() - j_minus) / (xnu * PI).sin();
            let yp = (jp * (xnu * PI).cos() - jp_minus) / (xnu * PI).sin();
            Ok((j, y, jp, yp))
        }
    } else {
        // For large x, use asymptotic expansion
        let (j, jp) = bessj_asymptotic(x, xnu);
        let (y, yp) = bessy_asymptotic(x, xnu);
        Ok((j, y, jp, yp))
    }
}

/// Spherical Bessel functions j_n(x), y_n(x) and their derivatives
pub fn sphbes(n: i32, x: f64) -> Result<(f64, f64, f64, f64), String> {
    if n < 0 || x <= 0.0 {
        return Err("n must be >= 0 and x must be > 0".to_string());
    }

    let order = n as f64 + 0.5;
    let (rj, ry, rjp, ryp) = bessjy(x, order)?;
    
    let factor = RTPIO2 / x.sqrt();
    
    let sj = factor * rj;
    let sy = factor * ry;
    let sjp = factor * rjp - sj / (2.0 * x);
    let syp = factor * ryp - sy / (2.0 * x);
    
    Ok((sj, sy, sjp, syp))
}

/// Individual spherical Bessel function of the first kind j_n(x)
pub fn sphbes_j(n: i32, x: f64) -> Result<f64, String> {
    sphbes(n, x).map(|(sj, _, _, _)| sj)
}

/// Individual spherical Bessel function of the second kind y_n(x)
pub fn sphbes_y(n: i32, x: f64) -> Result<f64, String> {
    sphbes(n, x).map(|(_, sy, _, _)| sy)
}

/// Derivative of spherical Bessel function of the first kind j'_n(x)
pub fn sphbes_jp(n: i32, x: f64) -> Result<f64, String> {
    sphbes(n, x).map(|(_, _, sjp, _)| sjp)
}

/// Derivative of spherical Bessel function of the second kind y'_n(x)
pub fn sphbes_yp(n: i32, x: f64) -> Result<f64, String> {
    sphbes(n, x).map(|(_, _, _, syp)| syp)
}

/// Spherical Hankel functions of the first and second kind (real and imaginary parts)
pub fn sphbes_h1(n: i32, x: f64) -> Result<(f64, f64, f64, f64), String> {
    let (sj, sy, sjp, syp) = sphbes(n, x)?;
    // h1 = j + i*y
    Ok((sj, sy, sjp, syp))
}

pub fn sphbes_h2(n: i32, x: f64) -> Result<(f64, f64, f64, f64), String> {
    let (sj, sy, sjp, syp) = sphbes(n, x)?;
    // h2 = j - i*y
    Ok((sj, -sy, sjp, -syp))
}

/// Thread-safe cache for spherical Bessel function values
#[derive(Clone)]
pub struct SphericalBesselCache {
    cache: Arc<Mutex<std::collections::HashMap<(i32, u64), (f64, f64, f64, f64)>>>,
}

impl SphericalBesselCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get spherical Bessel values with caching
    pub fn get(&self, n: i32, x: f64) -> Result<(f64, f64, f64, f64), String> {
        if n < 0 || x <= 0.0 {
            return Err("Invalid arguments".to_string());
        }

        // Use quantized x values for caching to avoid floating point precision issues
        let x_quantized = (x * 1_000_000.0).round() as u64;
        let key = (n, x_quantized);
        
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&result) = cache.get(&key) {
                return Ok(result);
            }
        }

        let result = sphbes(n, x)?;
        
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, result);
        }
        
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

/// Parallel batch computation for multiple orders at fixed x
pub fn sphbes_batch_n_parallel(n_values: &[i32], x: f64) -> Vec<Result<(f64, f64, f64, f64), String>> {
    n_values.par_iter()
        .map(|&n| sphbes(n, x))
        .collect()
}

/// Parallel batch computation for multiple x values at fixed order
pub fn sphbes_batch_x_parallel(n: i32, x_values: &[f64]) -> Vec<Result<(f64, f64, f64, f64), String>> {
    x_values.par_iter()
        .map(|&x| sphbes(n, x))
        .collect()
}

/// Compute spherical Bessel functions for a grid of values
pub fn sphbes_grid_parallel(n_values: &[i32], x_values: &[f64]) -> Vec<Vec<Result<(f64, f64, f64, f64), String>>> {
    n_values.par_iter()
        .map(|&n| {
            x_values.iter()
                .map(|&x| sphbes(n, x))
                .collect()
        })
        .collect()
}

/// Compute the ratio j_n(x)/j_{n-1}(x)
pub fn sphbes_j_ratio(n: i32, x: f64) -> Result<f64, String> {
    if n <= 0 {
        return Err("n must be > 0 for ratio computation".to_string());
    }
    let j_n = sphbes_j(n, x)?;
    let j_n_minus_1 = sphbes_j(n - 1, x)?;
    
    if j_n_minus_1.abs() < 1e-12 {
        Err("Division by zero near zeros of j_{n-1}".to_string())
    } else {
        Ok(j_n / j_n_minus_1)
    }
}

/// Compute the ratio y_n(x)/y_{n-1}(x)
pub fn sphbes_y_ratio(n: i32, x: f64) -> Result<f64, String> {
    if n <= 0 {
        return Err("n must be > 0 for ratio computation".to_string());
    }
    let y_n = sphbes_y(n, x)?;
    let y_n_minus_1 = sphbes_y(n - 1, x)?;
    
    if y_n_minus_1.abs() < 1e-12 {
        Err("Division by zero near zeros of y_{n-1}".to_string())
    } else {
        Ok(y_n / y_n_minus_1)
    }
}

/// Compute the Wronskian: j_n(x)y'_n(x) - j'_n(x)y_n(x) = 1/x²
pub fn sphbes_wronskian(n: i32, x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        return Err("x must be positive".to_string());
    }
    
    let (j_n, y_n, jp_n, yp_n) = sphbes(n, x)?;
    let wronskian = j_n * yp_n - jp_n * y_n;
    let expected = 1.0 / (x * x);
    
    // Return the actual wronskian value
    Ok(wronskian)
}

/// Riccati-Bessel functions: x j_n(x) and x y_n(x)
pub fn riccati_bessel(n: i32, x: f64) -> Result<(f64, f64), String> {
    let (j_n, y_n, _, _) = sphbes(n, x)?;
    Ok((x * j_n, x * y_n))
}

/// Derivatives of Riccati-Bessel functions
pub fn riccati_bessel_derivatives(n: i32, x: f64) -> Result<(f64, f64), String> {
    let (j_n, y_n, jp_n, yp_n) = sphbes(n, x)?;
    let psi_n = x * j_n;
    let chi_n = x * y_n;
    let psi_p = j_n + x * jp_n;
    let chi_p = y_n + x * yp_n;
    Ok((psi_p, chi_p))
}

/// Compute zeros of spherical Bessel functions using McMahon's formula
pub fn sphbes_j_zeros(n: i32, k: usize) -> Vec<f64> {
    let n_f64 = n as f64;
    (1..=k)
        .map(|i| {
            let beta = (i as f64 + n_f64 / 2.0 - 0.25) * PI;
            let mu = 4.0 * n_f64 * n_f64;
            beta - (mu - 1.0) / (8.0 * beta) - (4.0 * (mu - 1.0) * (7.0 * mu - 31.0)) / (384.0 * beta.powi(3))
        })
        .collect()
}

/// Compute zeros of spherical Neumann functions
pub fn sphbes_y_zeros(n: i32, k: usize) -> Vec<f64> {
    let n_f64 = n as f64;
    (1..=k)
        .map(|i| {
            let beta = (i as f64 + n_f64 / 2.0 - 0.75) * PI;
            let mu = 4.0 * n_f64 * n_f64;
            beta - (mu - 1.0) / (8.0 * beta) - (4.0 * (mu - 1.0) * (7.0 * mu - 31.0)) / (384.0 * beta.powi(3))
        })
        .collect()
}

/// Asymptotic expansion for large x
pub fn sphbes_asymptotic_large_x(n: i32, x: f64) -> (f64, f64, f64, f64) {
    let n_f64 = n as f64;
    let phase = x - (n_f64 + 0.5) * PI / 2.0;
    let (sin_phase, cos_phase) = phase.sin_cos();
    
    let amplitude = 1.0 / x;
    let j = amplitude * cos_phase;
    let y = amplitude * sin_phase;
    let jp = -amplitude * sin_phase - amplitude * cos_phase / x;
    let yp = amplitude * cos_phase - amplitude * sin_phase / x;
    
    (j, y, jp, yp)
}

/// Asymptotic expansion for small x
pub fn sphbes_asymptotic_small_x(n: i32, x: f64) -> (f64, f64, f64, f64) {
    let n_f64 = n as f64;
    
    if n == 0 {
        let j = 1.0 - x * x / 6.0;
        let y = -1.0 / x - x / 2.0 * (x.ln() + 0.57721566490153286060);
        let jp = -x / 3.0;
        let yp = 1.0 / (x * x) - 0.5 * (x.ln() + 0.57721566490153286060 + 1.0);
        (j, y, jp, yp)
    } else {
        let factorial = gamma(n_f64 + 1.0);
        let j = x.powi(n) / ((2.0 * n_f64 + 1.0) * factorial) * (1.0 - x * x / (2.0 * (2.0 * n_f64 + 3.0)));
        let y = -((2.0 * n_f64 - 1.0) as f64) * factorial / PI * x.powi(-n - 1) * (1.0 + x * x / (2.0 * (1.0 - 2.0 * n_f64)));
        
        let jp = n_f64 * x.powi(n - 1) / ((2.0 * n_f64 + 1.0) * factorial);
        let yp = (n_f64 + 1.0) * (2.0 * n_f64 - 1.0) * factorial / PI * x.powi(-n - 2);
        
        (j, y, jp, yp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sphbes_basic() {
        // Test known values for n = 0
        let (j0, y0, jp0, yp0) = sphbes(0, 1.0).unwrap();
        
        // j0(x) = sin(x)/x, y0(x) = -cos(x)/x
        assert_relative_eq!(j0, 1.0f64.sin() / 1.0, epsilon = 1e-10);
        assert_relative_eq!(y0, -1.0f64.cos() / 1.0, epsilon = 1e-8); // Relaxed epsilon for y0
        
        // Test derivatives
        let jp0_exact = (1.0f64.cos() / 1.0 - 1.0f64.sin() / (1.0 * 1.0));
        let yp0_exact = (1.0f64.sin() / 1.0 + 1.0f64.cos() / (1.0 * 1.0));
        assert_relative_eq!(jp0, jp0_exact, epsilon = 1e-10);
        assert_relative_eq!(yp0, yp0_exact, epsilon = 1e-8);
    }

    #[test]
    fn test_sphbes_n1() {
        // Test n = 1
        let (j1, y1, jp1, yp1) = sphbes(1, 1.0).unwrap();
        
        // j1(x) = sin(x)/x² - cos(x)/x
        let j1_exact = 1.0f64.sin() / 1.0.powi(2) - 1.0f64.cos() / 1.0;
        assert_relative_eq!(j1, j1_exact, epsilon = 1e-10);
        
        // y1(x) = -cos(x)/x² - sin(x)/x
        let y1_exact = -1.0f64.cos() / 1.0.powi(2) - 1.0f64.sin() / 1.0;
        assert_relative_eq!(y1, y1_exact, epsilon = 1e-8);
    }

    #[test]
    fn test_sphbes_error_cases() {
        assert!(sphbes(-1, 1.0).is_err());
        assert!(sphbes(0, 0.0).is_err());
        assert!(sphbes(0, -1.0).is_err());
    }

    #[test]
    fn test_individual_functions() {
        let j0 = sphbes_j(0, 1.0).unwrap();
        let y0 = sphbes_y(0, 1.0).unwrap();
        let jp0 = sphbes_jp(0, 1.0).unwrap();
        let yp0 = sphbes_yp(0, 1.0).unwrap();
        
        assert_relative_eq!(j0, 1.0f64.sin() / 1.0, epsilon = 1e-10);
        assert_relative_eq!(y0, -1.0f64.cos() / 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ratios() {
        let ratio_j = sphbes_j_ratio(1, 1.0).unwrap();
        let j1 = sphbes_j(1, 1.0).unwrap();
        let j0 = sphbes_j(0, 1.0).unwrap();
        
        assert_relative_eq!(ratio_j, j1 / j0, epsilon = 1e-10);
    }

    #[test]
    fn test_wronskian() {
        let wronskian = sphbes_wronskian(0, 1.0).unwrap();
        let expected = 1.0 / (1.0 * 1.0);
        assert_relative_eq!(wronskian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_riccati_bessel() {
        let (psi, chi) = riccati_bessel(0, 1.0).unwrap();
        assert_relative_eq!(psi, 1.0f64.sin(), epsilon = 1e-10);
        assert_relative_eq!(chi, -1.0f64.cos(), epsilon = 1e-8);
    }

    #[test]
    fn test_zeros() {
        let zeros = sphbes_j_zeros(0, 3);
        // First zeros of j0(x): π, 2π, 3π, ...
        let expected = [PI, 2.0 * PI, 3.0 * PI];
        
        for (i, &zero) in zeros.iter().enumerate() {
            assert_relative_eq!(zero, expected[i], epsilon = 0.1); // McMahon's formula is approximate
        }
    }

    #[test]
    fn test_parallel_batch() {
        let n_values = [0, 1, 2];
        let x = 1.0;
        
        let results = sphbes_batch_n_parallel(&n_values, x);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = sphbes(n_values[i], x).unwrap();
            assert_relative_eq!(result.as_ref().unwrap().0, expected.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = SphericalBesselCache::new();
        
        let result1 = cache.get(0, 1.0).unwrap();
        let result2 = cache.get(0, 1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1.0, result2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.1, result2.1, epsilon = 1e-10);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_asymptotic_expansions() {
        // Test large x asymptotic
        let (j_asym, y_asym, jp_asym, yp_asym) = sphbes_asymptotic_large_x(2, 10.0);
        let (j_exact, y_exact, jp_exact, yp_exact) = sphbes(2, 10.0).unwrap();
        
        assert_relative_eq!(j_asym, j_exact, epsilon = 0.01);
        assert_relative_eq!(y_asym, y_exact, epsilon = 0.01);
        
        // Test small x asymptotic
        let (j_asym_small, y_asym_small, jp_asym_small, yp_asym_small) = sphbes_asymptotic_small_x(1, 0.1);
        let (j_exact_small, y_exact_small, jp_exact_small, yp_exact_small) = sphbes(1, 0.1).unwrap();
        
        assert_relative_eq!(j_asym_small, j_exact_small, epsilon = 0.01);
    }
}
