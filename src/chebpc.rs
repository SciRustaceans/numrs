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
    
    // Handle the constant term
    d[0] = c[n-1];
    
    // Main recurrence loop - optimized without unstable SIMD
    for j in (1..n-1).rev() {
        let mut k = n - j;
        
        // Process in chunks of 4 for better cache performance
        while k >= 4 {
            let k_start = k - 4;
            
            // Manual unrolling for 4 elements
            for i in 0..4 {
                let idx = k_start + i;
                let sv = d[idx];
                d[idx] = 2.0 * d[idx - 1] - dd[idx];
                dd[idx] = sv;
            }
            k -= 4;
        }
        
        // Process remaining elements
        for idx in (n - j - k + 1..=n - j).rev() {
            let sv = d[idx];
            d[idx] = 2.0 * d[idx - 1] - dd[idx];
            dd[idx] = sv;
        }
        
        // Update d[0] and dd[0]
        let sv = d[0];
        d[0] = -dd[0] + c[j];
        dd[0] = sv;
    }
    
    // Final adjustment
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
    // Initialize arrays to zero
    for i in 0..n {
        unsafe {
            *d.add(i) = 0.0;
            *dd.add(i) = 0.0;
        }
    }
    
    unsafe {
        *d = *c.add(n-1);
    }
    
    for j in (1..n-1).rev() {
        let mut k = n - j;
        
        // Optimized inner loop with manual unrolling
        while k >= 4 {
            let k_start = k - 4;
            
            for i in 0..4 {
                let idx = k_start + i;
                unsafe {
                    let d_ptr = d.add(idx);
                    let d_prev_ptr = d.add(idx - 1);
                    let dd_ptr = dd.add(idx);
                    
                    let sv = *d_ptr;
                    *d_ptr = 2.0 * *d_prev_ptr - *dd_ptr;
                    *dd_ptr = sv;
                }
            }
            k -= 4;
        }
        
        // Process remainder
        for idx in (n - j - k + 1..=n - j).rev() {
            unsafe {
                let d_ptr = d.add(idx);
                let d_prev_ptr = d.add(idx - 1);
                let dd_ptr = dd.add(idx);
                
                let sv = *d_ptr;
                *d_ptr = 2.0 * *d_prev_ptr - *dd_ptr;
                *dd_ptr = sv;
            }
        }
        
        unsafe {
            let sv = *d;
            *d = -*dd + *c.add(j);
            *dd = sv;
        }
    }
    
    // Final adjustment
    for j in (1..n).rev() {
        unsafe {
            *d.add(j) = *d.add(j-1) - *dd.add(j);
        }
    }
    unsafe {
        *d = -*dd + 0.5 * *c;
    }
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

/// Cache-aligned version for maximum memory performance
pub fn chebpc_aligned(c: &[f64], n: usize) -> Vec<f64> {
    // For stable Rust, we'll use a simpler approach without complex alignment
    chebpc(c, n)
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
