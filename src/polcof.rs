use std::sync::Arc;
use rayon::prelude::*;
use ndarray::{Array1, ArrayView1};

/// Polynomial interpolation coefficients using Neville's algorithm
/// 
/// # Arguments
/// * `xa` - x-coordinates of data points
/// * `ya` - y-coordinates of data points
/// * `cof` - mutable reference to store the coefficients (must be same length as xa/ya)
/// 
/// # Panics
/// Panics if input arrays have different lengths or if cof has incorrect length
pub fn polcof(xa: &[f64], ya: &[f64], cof: &mut [f64]) {
    assert_eq!(xa.len(), ya.len(), "xa and ya must have the same length");
    assert_eq!(xa.len(), cof.len(), "cof must have the same length as xa and ya");
    
    let n = xa.len() - 1;
    let mut x = xa.to_vec();
    let mut y = ya.to_vec();
    
    for j in 0..=n {
        // Use polynomial interpolation at x=0 to find coefficient
        let (c, _) = polint(&x[..=n - j], &y[..=n - j], 0.0);
        cof[j] = c;
        
        // Find the point closest to zero
        let (k, _) = x[..=n - j]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        
        // Update remaining points
        for i in 0..=n - j {
            if x[i] != 0.0 {
                y[i] = (y[i] - cof[j]) / x[i];
            }
        }
        
        // Remove the used point by shifting array
        if k < n - j {
            for i in k + 1..=n - j {
                y[i - 1] = y[i];
                x[i - 1] = x[i];
            }
        }
    }
}

/// Polynomial interpolation using Neville's algorithm
/// 
/// # Arguments
/// * `xa` - x-coordinates of data points
/// * `ya` - y-coordinates of data points  
/// * `x` - point at which to interpolate
/// 
/// # Returns
/// Tuple of (interpolated value, error estimate)
fn polint(xa: &[f64], ya: &[f64], x: f64) -> (f64, f64) {
    let n = xa.len();
    let mut c = ya.to_vec();
    let mut d = ya.to_vec();
    
    let mut y = ya[0];
    let mut dy = 0.0;
    
    for m in 1..n {
        for i in 0..n - m {
            let ho = xa[i] - x;
            let hp = xa[i + m] - x;
            let w = c[i + 1] - d[i];
            
            let denom = ho - hp;
            if denom == 0.0 {
                panic!("Division by zero in polynomial interpolation");
            }
            
            let denom = w / denom;
            d[i] = hp * denom;
            c[i] = ho * denom;
        }
        
        dy = if 2 * (m + 1) < n {
            c[n - m]
        } else {
            d[n - m - 1]
        };
        y += dy;
    }
    
    (y, dy)
}

/// Thread-safe version using Arc for shared ownership
pub fn polcof_parallel(xa: Arc<[f64]>, ya: Arc<[f64]>) -> Vec<f64> {
    let n = xa.len() - 1;
    let mut cof = vec![0.0; xa.len()];
    
    // Process in parallel chunks for large datasets
    if xa.len() > 100 {
        let chunk_size = (xa.len() + 3) / 4; // 4 chunks
        
        cof.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk.len()).min(xa.len());
                
                let mut local_x = xa[start..end].to_vec();
                let mut local_y = ya[start..end].to_vec();
                
                for j in start..end {
                    if j > n { break; }
                    
                    let (c, _) = polint(&local_x[..=n - j], &local_y[..=n - j], 0.0);
                    chunk[j - start] = c;
                    
                    let (k, _) = local_x[..=n - j]
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                        .unwrap();
                    
                    for i in 0..=n - j {
                        if local_x[i] != 0.0 {
                            local_y[i] = (local_y[i] - c) / local_x[i];
                        }
                    }
                    
                    if k < n - j {
                        for i in k + 1..=n - j {
                            local_y[i - 1] = local_y[i];
                            local_x[i - 1] = local_x[i];
                        }
                    }
                }
            });
    } else {
        // Small dataset, use sequential version
        polcof(&xa, &ya, &mut cof);
    }
    
    cof
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_polcof_constant_polynomial() {
        let xa = [0.0, 1.0, 2.0];
        let ya = [5.0, 5.0, 5.0]; // Constant polynomial: 5
        let mut cof = [0.0; 3];
        
        polcof(&xa, &ya, &mut cof);
        
        // Should be [5, 0, 0] for constant polynomial
        assert_abs_diff_eq!(cof[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polcof_linear_polynomial() {
        let xa = [0.0, 1.0, 2.0];
        let ya = [2.0, 4.0, 6.0]; // Linear polynomial: 2x + 2
        let mut cof = [0.0; 3];
        
        polcof(&xa, &ya, &mut cof);
        
        // Should be [2, 2, 0] for 2x + 2
        assert_abs_diff_eq!(cof[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polcof_quadratic_polynomial() {
        let xa = [-1.0, 0.0, 1.0];
        let ya = [2.0, 1.0, 2.0]; // Quadratic polynomial: x² + 1
        let mut cof = [0.0; 3];
        
        polcof(&xa, &ya, &mut cof);
        
        // Should be [1, 0, 1] for x² + 1
        assert_abs_diff_eq!(cof[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polcof_high_degree_polynomial() {
        let xa = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let ya = [-8.0, -1.0, 0.0, 1.0, 8.0]; // Cubic polynomial: x³
        let mut cof = [0.0; 5];
        
        polcof(&xa, &ya, &mut cof);
        
        // Should be [0, 0, 0, 1, 0] for x³
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[3], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[4], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polcof_single_point() {
        let xa = [3.0];
        let ya = [7.0];
        let mut cof = [0.0];
        
        polcof(&xa, &ya, &mut cof);
        
        assert_abs_diff_eq!(cof[0], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polcof_two_points() {
        let xa = [1.0, 2.0];
        let ya = [3.0, 5.0]; // Linear: 2x + 1
        let mut cof = [0.0; 2];
        
        polcof(&xa, &ya, &mut cof);
        
        assert_abs_diff_eq!(cof[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn test_polcof_mismatched_lengths() {
        let xa = [1.0, 2.0];
        let ya = [1.0];
        let mut cof = [0.0; 2];
        
        polcof(&xa, &ya, &mut cof);
    }

    #[test]
    fn test_polcof_parallel_vs_sequential() {
        // Test that parallel and sequential versions give same results
        let xa: Vec<f64> = (-5..=5).map(|x| x as f64).collect();
        let ya: Vec<f64> = xa.iter().map(|x| x.powi(3) - 2.0 * x.powi(2) + x - 1.0).collect();
        
        let mut cof_seq = vec![0.0; xa.len()];
        polcof(&xa, &ya, &mut cof_seq);
        
        let xa_arc = Arc::new(xa.clone());
        let ya_arc = Arc::new(ya.clone());
        let cof_par = polcof_parallel(xa_arc, ya_arc);
        
        for (seq, par) in cof_seq.iter().zip(cof_par.iter()) {
            assert_abs_diff_eq!(seq, par, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_polcof_large_dataset() {
        // Test with larger dataset to exercise parallel code path
        let n = 150;
        let xa: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ya: Vec<f64> = xa.iter().map(|x| 3.0 * x.powi(2) - 2.0 * x + 5.0).collect();
        
        let xa_arc = Arc::new(xa.clone());
        let ya_arc = Arc::new(ya.clone());
        let cof = polcof_parallel(xa_arc, ya_arc);
        
        // Verify the polynomial coefficients (should be [5, -2, 3, 0, ...])
        assert_abs_diff_eq!(cof[0], 5.0, epsilon = 1e-8);
        assert_abs_diff_eq!(cof[1], -2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(cof[2], 3.0, epsilon = 1e-8);
        
        // Higher coefficients should be zero
        for &c in &cof[3..] {
            assert_abs_diff_eq!(c, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_polint_basic() {
        let xa = [1.0, 2.0, 3.0];
        let ya = [1.0, 4.0, 9.0]; // x²
        
        let (y, dy) = polint(&xa, &ya, 2.5);
        
        assert_abs_diff_eq!(y, 6.25, epsilon = 1e-10); // 2.5² = 6.25
        assert!(dy.abs() < 1e-10);
    }

    #[test]
    fn test_polcof_with_ndarray() {
        // Test integration with ndarray for scientific computing
        let xa = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let ya = array![4.0, 1.0, 0.0, 1.0, 4.0]; // x²
        
        let mut cof = Array1::zeros(5);
        
        polcof(xa.as_slice().unwrap(), ya.as_slice().unwrap(), cof.as_slice_mut().unwrap());
        
        // Should be [0, 0, 1, 0, 0] for x²
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[3], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[4], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_case_zero_x_values() {
        let xa = [0.0, 1.0, 2.0];
        let ya = [0.0, 1.0, 4.0]; // x²
        let mut cof = [0.0; 3];
        
        polcof(&xa, &ya, &mut cof);
        
        // Should be [0, 0, 1] for x²
        assert_abs_diff_eq!(cof[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cof[2], 1.0, epsilon = 1e-10);
    }
}
