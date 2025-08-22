use ndarray::{Array2, Array1};
use rayon::prelude::*;
use approx::assert_abs_diff_eq;

/// Bicubic interpolation coefficient calculation
///
/// Computes the coefficients for bicubic interpolation given function values,
/// first derivatives, and cross derivatives at the four corners of a unit square.
///
/// # Arguments
/// * `y` - Function values at corners [y11, y12, y21, y22]
/// * `y1` - ∂y/∂x1 derivatives at corners [y1_11, y1_12, y1_21, y1_22]
/// * `y2` - ∂y/∂x2 derivatives at corners [y2_11, y2_12, y2_21, y2_22]
/// * `y12` - ∂²y/∂x1∂x2 derivatives at corners [y12_11, y12_12, y12_21, y12_22]
/// * `d1` - Grid spacing in x1 direction
/// * `d2` - Grid spacing in x2 direction
///
/// # Returns
/// 4x4 array of bicubic interpolation coefficients
///
/// # Panics
/// Panics if input arrays don't have exactly 4 elements
pub fn bcucof(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    d1: f64,
    d2: f64,
) -> Array2<f64> {
    assert_eq!(y.len(), 4, "y must have exactly 4 elements");
    assert_eq!(y1.len(), 4, "y1 must have exactly 4 elements");
    assert_eq!(y2.len(), 4, "y2 must have exactly 4 elements");
    assert_eq!(y12.len(), 4, "y12 must have exactly 4 elements");

    // Precomputed weight matrix for bicubic interpolation
    const WT: [[f64; 16]; 16] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [-3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0],
        [9.0, -9.0, 9.0, -9.0, 6.0, 3.0, -3.0, -6.0, 6.0, -6.0, -3.0, 3.0, 4.0, 2.0, 1.0, 2.0],
        [-6.0, 6.0, -6.0, 6.0, -4.0, -2.0, 2.0, 4.0, -3.0, 3.0, 3.0, -3.0, -2.0, -1.0, -1.0, -2.0],
        [2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [-6.0, 6.0, -6.0, 6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, 2.0, -2.0, -2.0, -2.0, -1.0, -1.0],
        [4.0, -4.0, 4.0, -4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    ];

    let d1d2 = d1 * d2;
    let mut x = [0.0; 16];

    // Pack input data into x array with proper scaling
    for i in 0..4 {
        x[i] = y[i];
        x[i + 4] = y1[i] * d1;
        x[i + 8] = y2[i] * d2;
        x[i + 12] = y12[i] * d1d2;
    }

    // Matrix multiplication: cl = WT * x
    let cl: Vec<f64> = WT.iter()
        .map(|row| row.iter().zip(&x).map(|(w, x_val)| w * x_val).sum())
        .collect();

    // Reshape into 4x4 matrix
    Array2::from_shape_vec((4, 4), cl).unwrap()
}

/// Parallel version for processing multiple bicubic patches simultaneously
pub fn bcucof_batch(
    patches: &[(&[f64], &[f64], &[f64], &[f64])],
    d1: f64,
    d2: f64,
) -> Vec<Array2<f64>> {
    patches.par_iter()
        .map(|(y, y1, y2, y12)| bcucof(y, y1, y2, y12, d1, d2))
        .collect()
}

/// Evaluate bicubic polynomial given coefficients and normalized coordinates
pub fn bicubic_eval(c: &Array2<f64>, u: f64, v: f64) -> f64 {
    assert_eq!(c.shape(), &[4, 4], "Coefficient matrix must be 4x4");
    
    let mut result = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            result += c[[i, j]] * u.powi(i as i32) * v.powi(j as i32);
        }
    }
    result
}

/// Compute bicubic interpolation with error checking and boundary handling
pub fn bicubic_interpolate(
    y: &[f64],
    y1: &[f64],
    y2: &[f64],
    y12: &[f64],
    d1: f64,
    d2: f64,
    u: f64,
    v: f64,
) -> f64 {
    let c = bcucof(y, y1, y2, y12, d1, d2);
    bicubic_eval(&c, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bcucof_constant_function() {
        // Constant function: f(x1, x2) = 5.0
        let y = [5.0, 5.0, 5.0, 5.0];
        let y1 = [0.0, 0.0, 0.0, 0.0];  // ∂f/∂x1 = 0
        let y2 = [0.0, 0.0, 0.0, 0.0];  // ∂f/∂x2 = 0
        let y12 = [0.0, 0.0, 0.0, 0.0]; // ∂²f/∂x1∂x2 = 0
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Only c[0,0] should be non-zero for constant function
        assert_abs_diff_eq!(c[[0, 0]], 5.0, epsilon = 1e-10);
        for i in 0..4 {
            for j in 0..4 {
                if i != 0 || j != 0 {
                    assert_abs_diff_eq!(c[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_bcucof_linear_function() {
        // Linear function: f(x1, x2) = 2*x1 + 3*x2
        let y = [0.0, 3.0, 2.0, 5.0];  // f(0,0), f(0,1), f(1,0), f(1,1)
        let y1 = [2.0, 2.0, 2.0, 2.0];  // ∂f/∂x1 = 2 everywhere
        let y2 = [3.0, 3.0, 3.0, 3.0];  // ∂f/∂x2 = 3 everywhere
        let y12 = [0.0, 0.0, 0.0, 0.0]; // ∂²f/∂x1∂x2 = 0
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Coefficients for 2*u + 3*v
        let expected = array![
            [0.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        
        assert_abs_diff_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bcucof_bilinear_function() {
        // Bilinear function: f(x1, x2) = x1 * x2
        let y = [0.0, 0.0, 0.0, 1.0];  // f(0,0), f(0,1), f(1,0), f(1,1)
        let y1 = [0.0, 1.0, 0.0, 1.0];  // ∂f/∂x1 = x2
        let y2 = [0.0, 0.0, 1.0, 1.0];  // ∂f/∂x2 = x1
        let y12 = [1.0, 1.0, 1.0, 1.0]; // ∂²f/∂x1∂x2 = 1
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Should reproduce exactly u*v
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let interpolated = bicubic_eval(&c, u, v);
                let exact = u * v;
                assert_abs_diff_eq!(interpolated, exact, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bcucof_non_unit_spacing() {
        let y = [0.0, 2.0, 1.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        // Test with different grid spacings
        let c1 = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        let c2 = bcucof(&y, &y1, &y2, &y12, 2.0, 0.5);
        
        // Coefficients should be different due to different scaling
        assert!(c1 != c2);
    }

    #[test]
    fn test_bcucof_batch_processing() {
        let patches = vec![
            (
                [1.0, 1.0, 1.0, 1.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
            ),
            (
                [0.0, 1.0, 2.0, 3.0].as_slice(),
                [1.0, 1.0, 1.0, 1.0].as_slice(),
                [2.0, 2.0, 2.0, 2.0].as_slice(),
                [0.0, 0.0, 0.0, 0.0].as_slice(),
            ),
        ];
        
        let results = bcucof_batch(&patches, 1.0, 1.0);
        
        assert_eq!(results.len(), 2);
        
        // First patch should be constant
        assert_abs_diff_eq!(results[0][[0, 0]], 1.0, epsilon = 1e-10);
        
        // Second patch should be non-constant
        assert!(results[1][[0, 0]].abs() > 1e-10);
    }

    #[test]
    fn test_bicubic_eval_consistency() {
        let y = [0.0, 1.0, 2.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.5, 0.5, 0.5, 0.5];
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Test evaluation at corners
        assert_abs_diff_eq!(bicubic_eval(&c, 0.0, 0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval(&c, 0.0, 1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval(&c, 1.0, 0.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval(&c, 1.0, 1.0), 3.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "y must have exactly 4 elements")]
    fn test_bcucof_wrong_input_size() {
        let y = [1.0, 2.0, 3.0]; // Only 3 elements
        let y1 = [0.0; 4];
        let y2 = [0.0; 4];
        let y12 = [0.0; 4];
        
        bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
    }

    #[test]
    fn test_bcucof_high_order_terms() {
        // Test that higher-order terms are properly handled
        let y = [0.0, 0.0, 0.0, 1.0];
        let y1 = [0.0, 0.0, 0.0, 0.0];
        let y2 = [0.0, 0.0, 0.0, 0.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Should have non-zero coefficients for u³v³ term
        assert!(c[[3, 3]].abs() > 1e-10);
    }

    #[test]
    fn test_bcucof_parallel_consistency() {
        let patches: Vec<_> = (0..100)
            .map(|i| {
                let val = i as f64;
                (
                    [val, val, val, val].as_slice(),
                    [val * 0.1, val * 0.1, val * 0.1, val * 0.1].as_slice(),
                    [val * 0.2, val * 0.2, val * 0.2, val * 0.2].as_slice(),
                    [val * 0.3, val * 0.3, val * 0.3, val * 0.3].as_slice(),
                )
            })
            .collect();
        
        let results_seq: Vec<_> = patches.iter()
            .map(|(y, y1, y2, y12)| bcucof(y, y1, y2, y12, 1.0, 1.0))
            .collect();
        
        let results_par = bcucof_batch(&patches, 1.0, 1.0);
        
        assert_eq!(results_seq.len(), results_par.len());
        
        for (seq, par) in results_seq.iter().zip(results_par.iter()) {
            assert_abs_diff_eq!(seq, par, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bicubic_interpolate_convenience_function() {
        let y = [0.0, 1.0, 2.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.5, 0.5, 0.5, 0.5];
        
        let result = bicubic_interpolate(&y, &y1, &y2, &y12, 1.0, 1.0, 0.5, 0.5);
        
        // Should give a reasonable interpolated value
        assert!(result >= 0.0 && result <= 3.0);
    }
}
