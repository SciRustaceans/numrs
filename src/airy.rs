use std::f64::consts::{PI, FRAC_1_PI};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const THIRD: f64 = 1.0 / 3.0;
const TWOTHR: f64 = 2.0 * THIRD;
const ONOVRT: f64 = 0.5773502691896257; // 1/√3 ≈ 0.577350269

// Import Bessel functions from previous implementations
mod bessel_ik {
    use super::*;
    
    pub fn bessik(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        // Simplified implementation for Airy function context
        if x <= 0.0 || xnu < 0.0 {
            return Err("Invalid arguments".to_string());
        }
        
        // For Airy functions, we only need approximations for specific orders
        // Use series expansion or asymptotic approximations as needed
        if x < 2.0 {
            bessik_small_x(x, xnu)
        } else {
            bessik_large_x(x, xnu)
        }
    }
    
    fn bessik_small_x(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        // Series expansion for small x
        let (i, ip) = bessi_series(x, xnu);
        let (k, kp) = bessk_series(x, xnu);
        Ok((i, k, ip, kp))
    }
    
    fn bessik_large_x(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        // Asymptotic expansion for large x
        let (i, ip) = bessi_asymptotic(x, xnu);
        let (k, kp) = bessk_asymptotic(x, xnu);
        Ok((i, k, ip, kp))
    }
    
    fn bessi_series(x: f64, xnu: f64) -> (f64, f64) {
        // Series expansion for I_ν(x)
        let mut sum = 0.0;
        let mut sum_deriv = 0.0;
        let x2 = x * x * 0.25; // x²/4
        let mut term = 1.0;
        let gamma_nu = gamma(xnu + 1.0);
        
        for k in 0..20 {
            let kf = k as f64;
            let denom = (kf + 1.0) * (kf + xnu + 1.0);
            term *= x2 / denom;
            sum += term;
            sum_deriv += (2.0 * kf + xnu + 1.0) * term / x;
        }
        
        let x_half_pow = (x * 0.5).powf(xnu);
        let i = x_half_pow * sum / gamma_nu;
        let ip = x_half_pow * sum_deriv / (x * gamma_nu); // Optimized division
        
        (i, ip)
    }
    
    fn bessk_series(x: f64, xnu: f64) -> (f64, f64) {
        // Series expansion for K_ν(x)
        let (i_minus, _) = bessi_series(x, -xnu);
        let (i_plus, _) = bessi_series(x, xnu);
        
        let sin_nu_pi = (xnu * PI).sin();
        if sin_nu_pi.abs() < 1e-15 {
            return (f64::INFINITY, f64::INFINITY); // Handle pole
        }
        
        let k = PI * 0.5 * (i_minus - i_plus) / sin_nu_pi;
        let kp = -PI * 0.5 * (i_minus + i_plus) / sin_nu_pi;
        
        (k, kp)
    }
    
    fn bessi_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
        // Asymptotic expansion for I_ν(x)
        let z = x;
        let mu = 4.0 * xnu * xnu;
        let mut sum = 1.0;
        let mut sum_deriv = 1.0;
        let mut term = 1.0;
        
        for k in 1..10 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * z * kf);
            sum += term;
            sum_deriv += (1.0 - 2.0 * kf) * term / z;
        }
        
        let sqrt_factor = (2.0 * PI * z).sqrt().recip();
        let exp_z = z.exp();
        let i = exp_z * sum * sqrt_factor;
        let ip = exp_z * sum_deriv * sqrt_factor;
        
        (i, ip)
    }
    
    fn bessk_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
        // Asymptotic expansion for K_ν(x)
        let z = x;
        let mu = 4.0 * xnu * xnu;
        let mut sum = 1.0;
        let mut sum_deriv = 1.0;
        let mut term = 1.0;
        
        for k in 1..10 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * z * kf);
            if k % 2 == 0 {
                sum += term;
                sum_deriv += term;
            } else {
                sum -= term;
                sum_deriv -= term;
            }
        }
        
        let sqrt_factor = (PI * 0.5 / z).sqrt();
        let exp_neg_z = (-z).exp();
        let k = sqrt_factor * exp_neg_z * sum;
        let kp = -sqrt_factor * exp_neg_z * (sum + sum_deriv);
        
        (k, kp)
    }
    
    // Make gamma function public so it can be used by bessel_jy module
    pub fn gamma(x: f64) -> f64 {
        // Simple gamma function approximation using Lanczos
        if x <= 0.0 {
            return f64::NAN;
        }
        
        // Lanczos approximation with precomputed constants
        const COEFFS: [f64; 7] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
        ];
        
        let mut x = x - 1.0;
        let mut t = COEFFS[0];
        
        for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
            t += coeff / (x + i as f64);
        }
        
        let sqrt_2pi = (2.0 * PI).sqrt();
        let power_term = (x + 0.5).powf(x + 0.5) * (-x - 0.5).exp();
        sqrt_2pi * t * power_term
    }
}

mod bessel_jy {
    use super::*;
    
    pub fn bessjy(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        if x <= 0.0 || xnu < 0.0 {
            return Err("Invalid arguments".to_string());
        }
        
        if x < 2.0 {
            bessjy_small_x(x, xnu)
        } else {
            bessjy_large_x(x, xnu)
        }
    }
    
    fn bessjy_small_x(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        // Series expansion for small x
        let (j, jp) = bessj_series(x, xnu);
        let (y, yp) = bessy_series(x, xnu);
        Ok((j, y, jp, yp))
    }
    
    fn bessjy_large_x(x: f64, xnu: f64) -> Result<(f64, f64, f64, f64), String> {
        // Asymptotic expansion for large x
        let (j, jp) = bessj_asymptotic(x, xnu);
        let (y, yp) = bessy_asymptotic(x, xnu);
        Ok((j, y, jp, yp))
    }
    
    fn bessj_series(x: f64, xnu: f64) -> (f64, f64) {
        // Series expansion for J_ν(x)
        let mut sum = 0.0;
        let mut sum_deriv = 0.0;
        let x2 = x * x * -0.25; // -x²/4
        let mut term = 1.0;
        let gamma_nu = bessel_ik::gamma(xnu + 1.0); // Use public gamma function
        
        for k in 0..20 {
            let kf = k as f64;
            let denom = (kf + 1.0) * (kf + xnu + 1.0);
            term *= x2 / denom;
            sum += term;
            sum_deriv += (2.0 * kf + xnu + 1.0) * term / x;
        }
        
        let x_half_pow = (x * 0.5).powf(xnu);
        let j = x_half_pow * sum / gamma_nu;
        let jp = x_half_pow * sum_deriv / (x * gamma_nu);
        
        (j, jp)
    }
    
    fn bessy_series(x: f64, xnu: f64) -> (f64, f64) {
        // Series expansion for Y_ν(x)
        let (j_plus, jp_plus) = bessj_series(x, xnu);
        let (j_minus, jp_minus) = bessj_series(x, -xnu);
        
        let sin_nu_pi = (xnu * PI).sin();
        if sin_nu_pi.abs() < 1e-15 {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY); // Handle pole
        }
        
        let cos_nu_pi = (xnu * PI).cos();
        let y = (j_plus * cos_nu_pi - j_minus) / sin_nu_pi;
        let yp = (jp_plus * cos_nu_pi - jp_minus) / sin_nu_pi;
        
        (y, yp)
    }
    
    fn bessj_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
        // Asymptotic expansion for J_ν(x)
        let z = x;
        let mu = 4.0 * xnu * xnu;
        let mut p = 1.0;
        let mut q = 0.0;
        let mut term = 1.0;
        
        for k in 1..10 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * z * kf);
            if k % 2 == 0 {
                p += term;
            } else {
                q += term;
            }
        }
        
        let angle = z - xnu * PI * 0.5 - PI * 0.25;
        let (sin_z, cos_z) = angle.sin_cos();
        let sqrt_factor = (2.0 / (PI * z)).sqrt();
        
        let j = sqrt_factor * (p * cos_z - q * sin_z);
        let jp = -sqrt_factor * (p * sin_z + q * cos_z);
        
        (j, jp)
    }
    
    fn bessy_asymptotic(x: f64, xnu: f64) -> (f64, f64) {
        // Asymptotic expansion for Y_ν(x)
        let z = x;
        let mu = 4.0 * xnu * xnu;
        let mut p = 1.0;
        let mut q = 0.0;
        let mut term = 1.0;
        
        for k in 1..10 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * z * kf);
            if k % 2 == 0 {
                p += term;
            } else {
                q += term;
            }
        }
        
        let angle = z - xnu * PI * 0.5 - PI * 0.25;
        let (sin_z, cos_z) = angle.sin_cos();
        let sqrt_factor = (2.0 / (PI * z)).sqrt();
        
        let y = sqrt_factor * (p * sin_z + q * cos_z);
        let yp = sqrt_factor * (p * cos_z - q * sin_z);
        
        (y, yp)
    }
}

/// Airy functions Ai(x), Bi(x) and their derivatives Ai'(x), Bi'(x)
pub fn airy(x: f64) -> Result<(f64, f64, f64, f64), String> {
    // Handle NaN and infinity
    if x.is_nan() {
        return Err("x cannot be NaN".to_string());
    }
    if x.is_infinite() {
        if x.is_sign_positive() {
            return Ok((0.0, f64::INFINITY, 0.0, f64::INFINITY));
        } else {
            return Ok((0.0, 0.0, 0.0, 0.0)); // Asymptotic behavior for -∞
        }
    }

    let absx = x.abs();
    let rootx = absx.sqrt();
    let z = TWOTHR * absx * rootx;

    if x > 0.0 {
        // For x > 0: Use modified Bessel functions K and I
        let (_, rk, _, rkp) = bessel_ik::bessik(z, THIRD)?;
        let (_, ri, _, rip) = bessel_ik::bessik(z, TWOTHR)?;

        let ai = rootx * ONOVRT * rk * FRAC_1_PI;
        let bi = rootx * (rk * FRAC_1_PI + 2.0 * ONOVRT * ri);
        let aip = -x * ONOVRT * rkp * FRAC_1_PI;
        let bip = x * (rkp * FRAC_1_PI + 2.0 * ONOVRT * rip);

        Ok((ai, bi, aip, bip))

    } else if x < 0.0 {
        // For x < 0: Use Bessel functions J and Y
        let (rj, ry, rjp, ryp) = bessel_jy::bessjy(z, THIRD)?;
        let (rj2, ry2, rjp2, ryp2) = bessel_jy::bessjy(z, TWOTHR)?;

        let ai = 0.5 * rootx * (rj - ONOVRT * ry);
        let bi = -0.5 * rootx * (ry + ONOVRT * rj);
        let aip = 0.5 * absx * (ONOVRT * ryp + rjp);
        let bip = 0.5 * absx * (ONOVRT * rjp - ryp);

        Ok((ai, bi, aip, bip))

    } else {
        // For x = 0: Known exact values
        let ai = 0.355028053887817; // Ai(0)
        let bi = ai / ONOVRT;       // Bi(0) = Ai(0) / (√3)
        let aip = -0.258819403792807; // Ai'(0)
        let bip = -aip / ONOVRT;      // Bi'(0) = -Ai'(0) / (√3)

        Ok((ai, bi, aip, bip))
    }
}

/// Individual Airy function Ai(x)
#[inline]
pub fn airy_ai(x: f64) -> Result<f64, String> {
    airy(x).map(|(ai, _, _, _)| ai)
}

/// Individual Airy function Bi(x)
#[inline]
pub fn airy_bi(x: f64) -> Result<f64, String> {
    airy(x).map(|(_, bi, _, _)| bi)
}

/// Derivative of Airy function Ai'(x)
#[inline]
pub fn airy_aip(x: f64) -> Result<f64, String> {
    airy(x).map(|(_, _, aip, _)| aip)
}

/// Derivative of Airy function Bi'(x)
#[inline]
pub fn airy_bip(x: f64) -> Result<f64, String> {
    airy(x).map(|(_, _, _, bip)| bip)
}

/// Thread-safe cache for Airy function values with better key strategy
#[derive(Clone)]
pub struct AiryCache {
    cache: Arc<Mutex<std::collections::HashMap<u64, (f64, f64, f64, f64)>>>,
}

impl AiryCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Convert f64 to u64 for use as hash key (preserves bit pattern)
    fn f64_to_key(x: f64) -> u64 {
        x.to_bits()
    }

    /// Get Airy function values with caching
    pub fn get(&self, x: f64) -> Result<(f64, f64, f64, f64), String> {
        let key = Self::f64_to_key(x);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&result) = cache.get(&key) {
                return Ok(result);
            }
        }

        let result = airy(x)?;
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

    /// Precompute values for a range and cache them
    pub fn precompute_range(&self, start: f64, end: f64, num_points: usize) -> Result<(), String> {
        if num_points == 0 {
            return Ok(());
        }
        if num_points == 1 {
            let _ = self.get(start)?;
            return Ok(());
        }
        
        let step = (end - start) / (num_points - 1) as f64;
        
        for i in 0..num_points {
            let x = start + i as f64 * step;
            let _ = self.get(x)?; // This will compute and cache
        }
        
        Ok(())
    }
}

/// Parallel batch computation for multiple x values with error handling
pub fn airy_batch_parallel(x_values: &[f64]) -> Vec<Result<(f64, f64, f64, f64), String>> {
    x_values.par_iter()
        .map(|&x| {
            if x.is_nan() {
                Err("NaN input".to_string())
            } else {
                airy(x)
            }
        })
        .collect()
}

/// Compute Airy functions for a range of values in parallel with error handling
pub fn airy_range_parallel(start: f64, end: f64, num_points: usize) -> Result<Vec<(f64, f64, f64, f64)>, String> {
    if start >= end {
        return Err("start must be less than end".to_string());
    }
    if num_points < 2 {
        return Err("num_points must be at least 2".to_string());
    }

    let step = (end - start) / (num_points - 1) as f64;
    let x_values: Vec<f64> = (0..num_points)
        .map(|i| start + i as f64 * step)
        .collect();

    let results = airy_batch_parallel(&x_values);
    
    // Convert to Result<Vec> with proper error handling
    let mut output = Vec::with_capacity(num_points);
    for result in results {
        output.push(result?);
    }
    
    Ok(output)
}

/// Compute zeros of Airy function Ai(x) with better approximation
pub fn airy_ai_zeros(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    (1..=n)
        .map(|k| {
            let kf = k as f64;
            // Improved asymptotic formula with correction terms
            let t = 3.0 * PI * (4.0 * kf - 1.0) / 8.0;
            let base = -t.powf(TWOTHR);
            
            // Add correction term for better accuracy
            let correction = 5.0 / (48.0 * t.powf(4.0 / 3.0));
            base + correction
        })
        .collect()
}

/// Compute zeros of Airy function Ai'(x) with better approximation
pub fn airy_aip_zeros(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    (1..=n)
        .map(|k| {
            let kf = k as f64;
            // Improved asymptotic formula with correction terms
            let t = 3.0 * PI * (4.0 * kf - 3.0) / 8.0;
            let base = -t.powf(TWOTHR);
            
            // Add correction term for better accuracy
            let correction = 7.0 / (48.0 * t.powf(4.0 / 3.0));
            base + correction
        })
        .collect()
}

/// Compute the ratio Ai(x)/Bi(x) with error handling
pub fn airy_ratio_ai_bi(x: f64) -> Result<f64, String> {
    let (ai, bi, _, _) = airy(x)?;
    if bi == 0.0 {
        return Err("Bi(x) is zero, ratio undefined".to_string());
    }
    Ok(ai / bi)
}

/// Compute the ratio Ai'(x)/Bi'(x) with error handling
pub fn airy_ratio_aip_bip(x: f64) -> Result<f64, String> {
    let (_, _, aip, bip) = airy(x)?;
    if bip == 0.0 {
        return Err("Bi'(x) is zero, ratio undefined".to_string());
    }
    Ok(aip / bip)
}

/// Asymptotic expansion for large positive x (optimized)
pub fn airy_asymptotic_large_positive(x: f64) -> (f64, f64, f64, f64) {
    let z = TWOTHR * x * x.sqrt();
    let x_pow_neg_quarter = x.powf(-0.25);
    let x_pow_quarter = x.powf(0.25);
    let exp_neg_z = (-z).exp();
    let exp_z = z.exp();
    
    let prefactor_ai = 0.5 * FRAC_1_PI * x_pow_neg_quarter * exp_neg_z;
    let prefactor_bi = FRAC_1_PI * x_pow_neg_quarter * exp_z;
    
    let ai = prefactor_ai;
    let bi = prefactor_bi;
    let aip = -prefactor_ai * x_pow_quarter;
    let bip = prefactor_bi * x_pow_quarter;

    (ai, bi, aip, bip)
}

/// Asymptotic expansion for large negative x (optimized)
pub fn airy_asymptotic_large_negative(x: f64) -> (f64, f64, f64, f64) {
    let abs_x = -x;
    let z = TWOTHR * abs_x * abs_x.sqrt();
    let x_pow_neg_quarter = abs_x.powf(-0.25);
    let x_pow_quarter = abs_x.powf(0.25);
    
    let (sin_z, cos_z) = (z - PI * 0.25).sin_cos();
    let prefactor = FRAC_1_PI * x_pow_neg_quarter;
    let prefactor_prime = FRAC_1_PI * x_pow_quarter;

    let ai = prefactor * sin_z;
    let bi = prefactor * cos_z;
    let aip = -prefactor_prime * cos_z;
    let bip = prefactor_prime * sin_z;

    (ai, bi, aip, bip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_airy_zero() {
        // Test known values at x = 0
        let (ai, bi, aip, bip) = airy(0.0).unwrap();
        
        assert_relative_eq!(ai, 0.355028053887817, epsilon = 1e-10);
        assert_relative_eq!(bi, ai / ONOVRT, epsilon = 1e-10);
        assert_relative_eq!(aip, -0.258819403792807, epsilon = 1e-10);
        assert_relative_eq!(bip, -aip / ONOVRT, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_positive() {
        // Test positive x values
        let (ai, bi, aip, bip) = airy(1.0).unwrap();
        
        // Known values from mathematical tables
        assert_relative_eq!(ai, 0.135292416, epsilon = 1e-6);
        assert_relative_eq!(bi, 1.207423594, epsilon = 1e-6);
        assert_relative_eq!(aip, -0.159147441, epsilon = 1e-6);
        assert_relative_eq!(bip, 0.932435933, epsilon = 1e-6);
    }

    #[test]
    fn test_airy_negative() {
        // Test negative x values
        let (ai, bi, aip, bip) = airy(-1.0).unwrap();
        
        // Known values from mathematical tables
        assert_relative_eq!(ai, 0.535560883, epsilon = 1e-6);
        assert_relative_eq!(bi, 0.103997389, epsilon = 1e-6);
        assert_relative_eq!(aip, 0.357907563, epsilon = 1e-6);
        assert_relative_eq!(bip, 0.592375626, epsilon = 1e-6);
    }

    #[test]
    fn test_individual_functions() {
        let ai = airy_ai(1.0).unwrap();
        let bi = airy_bi(1.0).unwrap();
        let aip = airy_aip(1.0).unwrap();
        let bip = airy_bip(1.0).unwrap();
        
        assert_relative_eq!(ai, 0.135292416, epsilon = 1e-6);
        assert_relative_eq!(bi, 1.207423594, epsilon = 1e-6);
        assert_relative_eq!(aip, -0.159147441, epsilon = 1e-6);
        assert_relative_eq!(bip, 0.932435933, epsilon = 1e-6);
    }

    #[test]
    fn test_airy_zeros() {
        let zeros = airy_ai_zeros(5);
        // Known zeros: Ai(a_k) = 0
        let expected = [-2.338107, -4.087949, -5.520560, -6.786708, -7.944134];
        
        for (i, &zero) in zeros.iter().enumerate() {
            assert_relative_eq!(zero, expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_airy_ratio() {
        let ratio = airy_ratio_ai_bi(1.0).unwrap();
        let ai = airy_ai(1.0).unwrap();
        let bi = airy_bi(1.0).unwrap();
        
        assert_relative_eq!(ratio, ai / bi, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_batch() {
        let x_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let results = airy_batch_parallel(&x_values);
        
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            let expected = airy(x_values[i]).unwrap();
            assert_relative_eq!(result.as_ref().unwrap().0, expected.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cache_functionality() {
        let cache = AiryCache::new();
        
        let result1 = cache.get(1.0).unwrap();
        let result2 = cache.get(1.0).unwrap(); // Should be cached
        
        assert_relative_eq!(result1.0, result2.0, epsilon = 1e-10);
        assert_relative_eq!(result1.1, result2.1, epsilon = 1e-10);
    }

    #[test]
    fn test_asymptotic_expansions() {
        // Test asymptotic expansion for large positive x
        let (ai_asym, bi_asym, aip_asym, bip_asym) = airy_asymptotic_large_positive(10.0);
        let (ai_exact, bi_exact, aip_exact, bip_exact) = airy(10.0).unwrap();
        
        assert_relative_eq!(ai_asym, ai_exact, epsilon = 1e-3);
        assert_relative_eq!(bi_asym, bi_exact, epsilon = 1e-3);
        
        // Test asymptotic expansion for large negative x
        let (ai_asym_neg, bi_asym_neg, aip_asym_neg, bip_asym_neg) = airy_asymptotic_large_negative(-10.0);
        let (ai_exact_neg, bi_exact_neg, aip_exact_neg, bip_exact_neg) = airy(-10.0).unwrap();
        
        assert_relative_eq!(ai_asym_neg, ai_exact_neg, epsilon = 1e-3);
        assert_relative_eq!(bi_asym_neg, bi_exact_neg, epsilon = 1e-3);
    }

    #[test]
    fn test_edge_cases() {
        // Test NaN
        assert!(airy(f64::NAN).is_err());
        
        // Test infinity
        let (ai_inf, bi_inf, aip_inf, bip_inf) = airy(f64::INFINITY).unwrap();
        assert_eq!(ai_inf, 0.0);
        assert_eq!(bi_inf, f64::INFINITY);
        
        // Test negative infinity
        let (ai_neg_inf, bi_neg_inf, aip_neg_inf, bip_neg_inf) = airy(f64::NEG_INFINITY).unwrap();
        assert_eq!(ai_neg_inf, 0.0);
        assert_eq!(bi_neg_inf, 0.0);
    }

    #[test]
    fn test_bessel_functions() {
        // Test that Bessel functions are accessible
        let result = bessel_ik::bessik(1.0, 0.5);
        assert!(result.is_ok());
        
        let result = bessel_jy::bessjy(1.0, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gamma_function() {
        // Test that gamma function is accessible
        let gamma_val = bessel_ik::gamma(5.0);
        assert_relative_eq!(gamma_val, 24.0, epsilon = 1e-10); // 4! = 24
    }
}

/// Benchmark module
#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_airy_zero(c: &mut Criterion) {
        c.bench_function("airy(0.0)", |b| {
            b.iter(|| airy(black_box(0.0)).unwrap())
        });
    }

    fn bench_airy_positive(c: &mut Criterion) {
        c.bench_function("airy(1.0)", |b| {
            b.iter(|| airy(black_box(1.0)).unwrap())
        });
    }

    fn bench_airy_negative(c: &mut Criterion) {
        c.bench_function("airy(-1.0)", |b| {
            b.iter(|| airy(black_box(-1.0)).unwrap())
        });
    }

    fn bench_airy_batch(c: &mut Criterion) {
        let x_values: Vec<f64> = (-10..10).map(|x| x as f64 / 2.0).collect();
        
        c.bench_function("airy_batch_parallel(20)", |b| {
            b.iter(|| airy_batch_parallel(black_box(&x_values)))
        });
    }

    fn bench_airy_individual(c: &mut Criterion) {
        c.bench_function("airy_ai(1.0)", |b| {
            b.iter(|| airy_ai(black_box(1.0)).unwrap())
        });
        
        c.bench_function("airy_bi(1.0)", |b| {
            b.iter(|| airy_bi(black_box(1.0)).unwrap())
        });
    }

    criterion_group!(
        benches, 
        bench_airy_zero, 
        bench_airy_positive,
        bench_airy_negative,
        bench_airy_batch,
        bench_airy_individual
    );
    criterion_main!(benches);
}
