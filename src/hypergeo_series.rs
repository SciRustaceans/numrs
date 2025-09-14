// lib.rs
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::Arc;

const MAX_ITERATIONS: usize = 1000;

#[derive(Debug, Clone, Copy)]
pub struct HypergeometricResult {
    pub series: Complex64,
    pub deriv: Complex64,
}

pub fn hypser(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> HypergeometricResult {
    let mut deriv = Complex64::new(0.0, 0.0);
    let mut fac = Complex64::new(1.0, 0.0);
    let mut temp = fac;
    let mut aa = a;
    let mut bb = b;
    let mut cc = c;

    for n in 1..=MAX_ITERATIONS {
        // Calculate next term: fac * (aa * bb) / cc
        fac = fac * (aa * bb) / cc;
        
        // Add to derivative
        deriv += fac;
        
        // Multiply by z/n for next series term
        fac = fac * (z / (n as f64));
        
        // Calculate new series value
        let series = temp + fac;
        
        // Check for convergence
        if series == temp {
            return HypergeometricResult { series, deriv };
        }
        
        temp = series;
        
        // Increment parameters
        aa += Complex64::new(1.0, 0.0);
        bb += Complex64::new(1.0, 0.0);
        cc += Complex64::new(1.0, 0.0);
    }

    panic!("Convergence failed in hypser after {} iterations", MAX_ITERATIONS);
}

// Parallel version for multiple inputs
pub fn hypser_parallel(
    a_params: &[Complex64],
    b_params: &[Complex64],
    c_params: &[Complex64],
    z_params: &[Complex64],
) -> Vec<HypergeometricResult> {
    assert_eq!(a_params.len(), b_params.len());
    assert_eq!(a_params.len(), c_params.len());
    assert_eq!(a_params.len(), z_params.len());

    a_params
        .par_iter()
        .zip(b_params.par_iter())
        .zip(c_params.par_iter())
        .zip(z_params.par_iter())
        .map(|(((a, b), c), z)| hypser(*a, *b, *c, *z))
        .collect()
}

// Precomputed version for fixed parameters a, b, c
pub struct HypserCache {
    a: Complex64,
    b: Complex64,
    c: Complex64,
}

impl HypserCache {
    pub fn new(a: Complex64, b: Complex64, c: Complex64) -> Self {
        Self { a, b, c }
    }

    pub fn compute(&self, z: Complex64) -> HypergeometricResult {
        hypser(self.a, self.b, self.c, z)
    }
}

// Batch computation with precomputation
pub fn hypser_batch(cache: Arc<HypserCache>, z_params: &[Complex64]) -> Vec<HypergeometricResult> {
    z_params
        .par_iter()
        .map(|z| cache.compute(*z))
        .collect()
}

// Optimized version with early termination and convergence threshold
pub fn hypser_optimized(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    tolerance: Option<f64>,
) -> HypergeometricResult {
    let tol = tolerance.unwrap_or(1e-12);
    let mut deriv = Complex64::new(0.0, 0.0);
    let mut fac = Complex64::new(1.0, 0.0);
    let mut temp = fac;
    let mut aa = a;
    let mut bb = b;
    let mut cc = c;

    for n in 1..=MAX_ITERATIONS {
        // Calculate next term
        fac = fac * (aa * bb) / cc;
        
        // Add to derivative
        deriv += fac;
        
        // Multiply by z/n for next series term
        fac = fac * (z / (n as f64));
        
        // Calculate new series value
        let series = temp + fac;
        
        // Check for convergence with tolerance
        if (series - temp).norm() < tol {
            return HypergeometricResult { series, deriv };
        }
        
        temp = series;
        
        // Increment parameters
        aa += Complex64::new(1.0, 0.0);
        bb += Complex64::new(1.0, 0.0);
        cc += Complex64::new(1.0, 0.0);
    }

    panic!("Convergence failed in hypser after {} iterations", MAX_ITERATIONS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_hypser() {
        let a = Complex64::new(1.0, 0.0);
        let b = Complex64::new(1.0, 0.0);
        let c = Complex64::new(2.0, 0.0);
        let z = Complex64::new(0.5, 0.0);
        
        let result = hypser(a, b, c, z);
        println!("Series: {}, Deriv: {}", result.series, result.deriv);
        
        // Test parallel version
        let a_params = vec![a, a, a];
        let b_params = vec![b, b, b];
        let c_params = vec![c, c, c];
        let z_params = vec![z, z * 0.5, z * 0.25];
        let results = hypser_parallel(&a_params, &b_params, &c_params, &z_params);
        
        // Test cached version
        let cache = Arc::new(HypserCache::new(a, b, c));
        let batch_results = hypser_batch(cache, &[z, z * 0.5, z * 0.25]);
    }
}
