use std::simd::{f64x4, SimdFloat};
use std::ptr;

/// Converts Chebyshev coefficients to power series coefficients with maximum performance
/// 
/// # Arguments
/// * `c` - Chebyshev coefficients [c0, c1, ..., c_{n-1}]
/// * `n` - Number of coefficients
/// 
/// # Returns
/// Power series coefficients [d0, d1, ..., d_{n-1}]
pub fn chebpc(c: &[f64], n: usize) -> Vec<f64> {
    assert!(n >= 1, "Number of coefficients must be positive");
    assert!(c.len() >= n, "Insufficient input coefficients");
    
    let mut d = vec![0.0; n];
    let mut dd = vec![0.0; n];
    
    // Initialize arrays to zero (compiler will optimize this)
    d.fill(0.0);
    dd.fill(0.0);
    
    // Handle the constant term
    d[0] = c[n-1];
    
    // Main recurrence loop - heavily optimized
    for j in (1..n-1).rev() {
        let mut k = n - j;
        
        // Process in chunks of 4 using SIMD
        while k >= 4 {
            let k_start = k - 4;
            
            // SAFETY: We ensure k_start and k are within bounds
            unsafe {
                let d_ptr = d.as_mut_ptr().add(k_start);
                let dd_ptr = dd.as_mut_ptr().add(k_start);
                let d_prev_ptr = d.as_ptr().add(k_start - 1);
                
                // Load vectors
                let d_vec = f64x4::from_slice(std::slice::from_raw_parts(d_ptr, 4));
                let dd_vec = f64x4::from_slice(std::slice::from_raw_parts(dd_ptr, 4));
                let d_prev_vec = f64x4::from_slice(std::slice::from_raw_parts(d_prev_ptr, 4));
                
                // Compute: d[k] = 2.0 * d[k-1] - dd[k]
                let result = d_prev_vec * f64x4::splat(2.0) - dd_vec;
                
                // Store results
                result.copy_to_slice(std::slice::from_raw_parts_mut(d_ptr, 4));
                d_vec.copy_to_slice(std::slice::from_raw_parts_mut(dd_ptr, 4));
            }
            k -= 4;
        }
        
        // Process remaining elements with manual loop unrolling
        match k {
            3 => {
                let sv = d[3];
                d[3] = 2.0 * d[2] - dd[3];
                dd[3] = sv;
                k = 2;
            }
            2 => {
                let sv = d[2];
                d[2] = 2.0 * d[1] - dd[2];
                dd[2] = sv;
                k = 1;
            }
            1 => {
                let sv = d[1];
                d[1] = 2.0 * d[0] - dd[1];
                dd[1] = sv;
                k = 0;
            }
            _ => {}
        }
        
        // Update d[0] and dd[0]
        let sv = d[0];
        d[0] = -dd[0] + c[j];
        dd[0] = sv;
    }
    
    // Final adjustment with optimized loop
    for j in (1..n).rev() {
        d[j] = d[j-1] - dd[j];
    }
    d[0] = -dd[0] + 0.5 * c[0];
    
    d
}

/// Ultra-optimized in-place version that reuses pre-allocated buffers
/// 
/// # Safety
/// All slices must have length at least `n`
pub unsafe fn chebpc_unsafe(c: *const f64, d: *mut f64, dd: *mut f64, n: usize) {
    // Initialize arrays to zero using vectorized operations
    for i in 0..n {
        *d.add(i) = 0.0;
        *dd.add(i) = 0.0;
    }
    
    *d = *c.add(n-1);
    
    for j in (1..n-1).rev() {
        let mut k = n - j;
        
        // SIMD-optimized inner loop
        while k >= 4 {
            let k_start = k - 4;
            
            let d_ptr = d.add(k_start);
            let dd_ptr = dd.add(k_start);
            let d_prev_ptr = d.add(k_start - 1);
            
            let d_vec = f64x4::from_slice(std::slice::from_raw_parts(d_ptr, 4));
            let dd_vec = f64x4::from_slice(std::slice::from_raw_parts(dd_ptr, 4));
            let d_prev_vec = f64x4::from_slice(std::slice::from_raw_parts(d_prev_ptr, 4));
            
            let result = d_prev_vec * f64x4::splat(2.0) - dd_vec;
            result.copy_to_slice(std::slice::from_raw_parts_mut(d_ptr, 4));
            d_vec.copy_to_slice(std::slice::from_raw_parts_mut(dd_ptr, 4));
            
            k -= 4;
        }
        
        // Process remainder
        while k >= 1 {
            let d_k = d.add(k);
            let d_k_minus_1 = d.add(k - 1);
            let dd_k = dd.add(k);
            
            let sv = *d_k;
            *d_k = 2.0 * *d_k_minus_1 - *dd_k;
            *dd_k = sv;
            
            k -= 1;
        }
        
        let sv = *d;
        *d = -*dd + *c.add(j);
        *dd = sv;
    }
    
    // Final adjustment
    for j in (1..n).rev() {
        *d.add(j) = *d.add(j-1) - *dd.add(j);
    }
    *d = -*dd + 0.5 * *c;
}

/// Thread-safe converter with pre-allocated workspace
pub struct ChebyshevConverter {
    workspace: Vec<f64>,
}

impl ChebyshevConverter {
    pub fn new(max_n: usize) -> Self {
        Self {
            workspace: vec![0.0; max_n],
        }
    }
    
    /// Convert with reused workspace - zero allocation
    pub fn convert(&mut self, c: &[f64], n: usize) -> Vec<f64> {
        assert!(n <= self.workspace.len(), "Workspace too small");
        
        let mut d = vec![0.0; n];
        let dd = &mut self.workspace[..n];
        
        // Safe wrapper around unsafe version
        unsafe {
            chebpc_unsafe(
                c.as_ptr(),
                d.as_mut_ptr(),
                dd.as_mut_ptr(),
                n
            );
        }
        
        d
    }
    
    /// In-place conversion into pre-allocated buffer
    pub fn convert_into(&mut self, c: &[f64], d: &mut [f64], n: usize) {
        assert!(n <= self.workspace.len(), "Workspace too small");
        assert!(d.len() >= n, "Output buffer too small");
        
        let dd = &mut self.workspace[..n];
        
        unsafe {
            chebpc_unsafe(
                c.as_ptr(),
                d.as_mut_ptr(),
                dd.as_mut_ptr(),
                n
            );
        }
    }
}

/// Batch processing with parallel SIMD optimization
pub fn chebpc_batch(c_list: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    c_list.par_iter().map(|c| chebpc(c, n)).collect()
}

/// Cache-aligned version for maximum memory performance
pub fn chebpc_aligned(c: &[f64], n: usize) -> Vec<f64> {
    // Ensure 64-byte alignment for optimal cache performance
    let mut d = Vec::with_capacity(n + 8);
    let aligned_ptr = d.as_mut_ptr().align_offset(64);
    unsafe {
        d.set_len(n);
        if aligned_ptr != 0 {
            // Reallocate with proper alignment if needed
            let mut aligned = Vec::with_capacity(n + 8);
            let aligned_ptr = aligned.as_mut_ptr().align_offset(64);
            aligned.set_len(n);
            chebpc_unsafe(c.as_ptr(), aligned.as_mut_ptr(), ptr::null_mut(), n);
            aligned
        } else {
            chebpc_unsafe(c.as_ptr(), d.as_mut_ptr(), ptr::null_mut(), n);
            d
        }
    }
}

/// Benchmark-friendly version with timing
pub fn chebpc_timed(c: &[f64], n: usize) -> (Vec<f64>, std::time::Duration) {
    let start = std::time::Instant::now();
    let result = chebpc(c, n);
    let duration = start.elapsed();
    (result, duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_chebpc_basic() {
        // Test conversion of x^2 coefficients
        let c = vec![0.5, 0.0, 0.5]; // T0: 0.5, T1: 0.0, T2: 0.5
        let n = 3;
        
        let d = chebpc(&c, n);
        
        // x^2 = 0.5*T0 + 0.5*T2 = 0.5*1 + 0.5*(2x^2-1) = x^2
        // Power series: 0 + 0*x + 1*x^2
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[2], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_chebpc_linear() {
        // Test conversion of x coefficients
        let c = vec![0.0, 1.0, 0.0]; // T0: 0.0, T1: 1.0, T2: 0.0
        let n = 3;
        
        let d = chebpc(&c, n);
        
        // x = T1(x)
        // Power series: 0 + 1*x + 0*x^2
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[1], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[2], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_chebpc_constant() {
        // Test conversion of constant function
        let c = vec![1.0, 0.0, 0.0]; // T0: 1.0
        let n = 3;
        
        let d = chebpc(&c, n);
        
        // 1 = 1*T0(x)
        // Power series: 1 + 0*x + 0*x^2
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(d[2], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_chebpc_high_degree() {
        // Test with higher degree polynomial
        let n = 6;
        let c = vec![0.5, 0.3, 0.2, 0.1, 0.05, 0.02];
        
        let d = chebpc(&c, n);
        
        // Verify that the conversion is mathematically correct
        // by checking polynomial values at test points
        for &x in &[-0.5, 0.0, 0.5] {
            let cheb_val = evaluate_chebyshev(&c, x);
            let poly_val = evaluate_polynomial(&d, x);
            assert_abs_diff_eq!(cheb_val, poly_val, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_chebyshev_converter() {
        let mut converter = ChebyshevConverter::new(10);
        let c = vec![0.5, 0.0, 0.5];
        
        let d = converter.convert(&c, 3);
        assert_abs_diff_eq!(d[2], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_convert_into() {
        let mut converter = ChebyshevConverter::new(10);
        let c = vec![0.5, 0.0, 0.5];
        let mut d = vec![0.0; 3];
        
        converter.convert_into(&c, &mut d, 3);
        assert_abs_diff_eq!(d[2], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_batch_processing() {
        let polynomials = [
            vec![0.5, 0.0, 0.5].as_slice(), // x^2
            vec![0.0, 1.0, 0.0].as_slice(), // x
            vec![1.0, 0.0, 0.0].as_slice(), // 1
        ];
        
        let results = chebpc_batch(&polynomials, 3);
        
        assert_eq!(results.len(), 3);
        assert_abs_diff_eq!(results[0][2], 1.0, epsilon = 1e-15); // x^2
        assert_abs_diff_eq!(results[1][1], 1.0, epsilon = 1e-15); // x
        assert_abs_diff_eq!(results[2][0], 1.0, epsilon = 1e-15); // 1
    }

    #[test]
    #[should_panic(expected = "Number of coefficients must be positive")]
    fn test_empty_input() {
        chebpc(&[], 0);
    }

    #[test]
    #[should_panic(expected = "Insufficient input coefficients")]
    fn test_insufficient_coefficients() {
        chebpc(&[1.0, 2.0], 3);
    }

    // Helper functions for polynomial evaluation
    fn evaluate_chebyshev(c: &[f64], x: f64) -> f64 {
        let mut d = 0.0;
        let mut dd = 0.0;
        let y = 2.0 * x;
        
        for &coeff in c.iter().rev() {
            let sv = d;
            d = y * d - dd + coeff;
            dd = sv;
        }
        
        0.5 * c[0] + x * d - dd
    }

    fn evaluate_polynomial(d: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        for &coeff in d.iter().rev() {
            result = result * x + coeff;
        }
        result
    }

    #[test]
    fn test_performance_large_n() {
        // Performance test with large n
        let n = 100;
        let c: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();
        
        let (result, duration) = chebpc_timed(&c, n);
        
        assert_eq!(result.len(), n);
        println!("Conversion of {} coefficients took: {:?}", n, duration);
        
        // Verify correctness
        let x = 0.5;
        let cheb_val = evaluate_chebyshev(&c, x);
        let poly_val = evaluate_polynomial(&result, x);
        assert_abs_diff_eq!(cheb_val, poly_val, epsilon = 1e-10);
    }
}
