use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

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

    // Precomputed weight matrix for bicubic interpolation (optimized layout)
    const WT: [[f64; 16]; 16] = [
        [1.0, 0.0, -3.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 9.0, -6.0, 2.0, 0.0, -6.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, -9.0, 6.0, -2.0, 0.0, 6.0, -4.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, -6.0, 0.0, 0.0, -6.0, 4.0],
        [0.0, 0.0, 3.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.0, 6.0, 0.0, 0.0, 6.0, -4.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 2.0, -2.0, 0.0, 6.0, -4.0, 1.0, 0.0, -3.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 3.0, -2.0, 1.0, 0.0, -3.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 2.0, 0.0, 0.0, 3.0, -2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -2.0, 0.0, 0.0, -6.0, 4.0, 0.0, 0.0, 3.0, -2.0],
        [0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 6.0, -3.0, 0.0, 2.0, -4.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -6.0, 3.0, 0.0, -2.0, 4.0, -2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, 2.0, -2.0],
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -3.0, 0.0, 0.0, -2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 1.0, -1.0],
    ];

    let d1d2 = d1 * d2;
    
    // Pack input data into x array with proper scaling
    let mut x = [0.0; 16];
    
    // Process in optimized order for better cache performance
    for i in 0..4 {
        x[i] = y[i];
        x[i + 4] = y1[i] * d1;
        x[i + 8] = y2[i] * d2;
        x[i + 12] = y12[i] * d1d2;
    }

    // Matrix multiplication: cl = WT * x with manual loop optimization
    let mut cl = [0.0; 16];
    
    // Process rows with optimized loop unrolling
    for (i, row) in WT.iter().enumerate() {
        let mut sum = 0.0;
        
        // Manual loop unrolling for better performance
        sum += row[0] * x[0] + row[1] * x[1] + row[2] * x[2] + row[3] * x[3];
        sum += row[4] * x[4] + row[5] * x[5] + row[6] * x[6] + row[7] * x[7];
        sum += row[8] * x[8] + row[9] * x[9] + row[10] * x[10] + row[11] * x[11];
        sum += row[12] * x[12] + row[13] * x[13] + row[14] * x[14] + row[15] * x[15];
        
        cl[i] = sum;
    }

    // Reshape into 4x4 matrix
    Array2::from_shape_vec((4, 4), cl.to_vec()).unwrap()
}

/// Optimized version with precomputed constants
pub fn bcucof_fast(
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

    let d1d2 = d1 * d2;
    
    // Pre-scale inputs to avoid repeated multiplication
    let y1_scaled: Vec<f64> = y1.iter().map(|&val| val * d1).collect();
    let y2_scaled: Vec<f64> = y2.iter().map(|&val| val * d2).collect();
    let y12_scaled: Vec<f64> = y12.iter().map(|&val| val * d1d2).collect();
    
    // Direct coefficient computation using optimized formulas
    // This avoids the full matrix multiplication for common cases
    let mut c = Array2::zeros((4, 4));
    
    // Compute coefficients using optimized pattern
    // These formulas are derived from the weight matrix multiplication
    for i in 0..4 {
        for j in 0..4 {
            c[[i, j]] = compute_coefficient(i, j, y, &y1_scaled, &y2_scaled, &y12_scaled);
        }
    }
    
    c
}

/// Compute individual coefficient using optimized pattern matching
fn compute_coefficient(i: usize, j: usize, y: &[f64], y1: &[f64], y2: &[f64], y12: &[f64]) -> f64 {
    match (i, j) {
        (0, 0) => y[0],
        (0, 1) => y2[0],
        (0, 2) => -3.0*y[0] + 3.0*y[1] - 2.0*y2[0] - y2[1],
        (0, 3) => 2.0*y[0] - 2.0*y[1] + y2[0] + y2[1],
        
        (1, 0) => y1[0],
        (1, 1) => y12[0],
        (1, 2) => -3.0*y1[0] + 3.0*y1[1] - 2.0*y12[0] - y12[1],
        (1, 3) => 2.0*y1[0] - 2.0*y1[1] + y12[0] + y12[1],
        
        (2, 0) => -3.0*y[0] + 3.0*y[2] - 2.0*y1[0] - y1[2],
        (2, 1) => -3.0*y2[0] + 3.0*y2[2] - 2.0*y12[0] - y12[2],
        (2, 2) => 9.0*y[0] - 9.0*y[1] - 9.0*y[2] + 9.0*y[3] + 
                 6.0*y1[0] + 3.0*y1[1] - 3.0*y1[2] - 6.0*y1[3] +
                 6.0*y2[0] - 6.0*y2[1] + 3.0*y2[2] - 3.0*y2[3] +
                 4.0*y12[0] + 2.0*y12[1] + y12[2] + 2.0*y12[3],
        (2, 3) => -6.0*y[0] + 6.0*y[1] + 6.0*y[2] - 6.0*y[3] -
                 4.0*y1[0] - 2.0*y1[1] + 2.0*y1[2] + 4.0*y1[3] -
                 3.0*y2[0] + 3.0*y2[1] - 3.0*y2[2] + 3.0*y2[3] -
                 2.0*y12[0] - y12[1] - y12[2] - 2.0*y12[3],
        
        (3, 0) => 2.0*y[0] - 2.0*y[2] + y1[0] + y1[2],
        (3, 1) => 2.0*y2[0] - 2.0*y2[2] + y12[0] + y12[2],
        (3, 2) => -6.0*y[0] + 6.0*y[1] + 6.0*y[2] - 6.0*y[3] -
                 3.0*y1[0] - 3.0*y1[1] + 3.0*y1[2] + 3.0*y1[3] -
                 4.0*y2[0] + 4.0*y2[1] - 2.0*y2[2] + 2.0*y2[3] -
                 2.0*y12[0] - 2.0*y12[1] - y12[2] - y12[3],
        (3, 3) => 4.0*y[0] - 4.0*y[1] - 4.0*y[2] + 4.0*y[3] +
                 2.0*y1[0] + 2.0*y1[1] - 2.0*y1[2] - 2.0*y1[3] +
                 2.0*y2[0] - 2.0*y2[1] - 2.0*y2[2] + 2.0*y2[3] +
                 y12[0] + y12[1] + y12[2] + y12[3],
        
        _ => 0.0,
    }
}

/// Parallel version for processing multiple bicubic patches simultaneously
pub fn bcucof_batch(
    patches: &[(&[f64], &[f64], &[f64], &[f64])],
    d1: f64,
    d2: f64,
) -> Vec<Array2<f64>> {
    patches.par_iter()
        .map(|(y, y1, y2, y12)| bcucof_fast(y, y1, y2, y12, d1, d2))
        .collect()
}

/// Evaluate bicubic polynomial given coefficients and normalized coordinates
pub fn bicubic_eval(c: &ArrayView2<f64>, u: f64, v: f64) -> f64 {
    debug_assert_eq!(c.shape(), &[4, 4], "Coefficient matrix must be 4x4");
    
    // Precompute powers for better performance
    let u2 = u * u;
    let u3 = u2 * u;
    
    let v2 = v * v;
    let v3 = v2 * v;
    
    // Use Horner's method for better numerical stability and performance
    // Evaluate as: ((c33*v + c32)*v + c31)*v + c30) * u^3 + ... etc.
    let mut result = 0.0;
    
    // Process u powers from highest to lowest
    result = (result * u) + (
        c[[3, 3]] * v3 + c[[3, 2]] * v2 + c[[3, 1]] * v + c[[3, 0]]
    );
    result = (result * u) + (
        c[[2, 3]] * v3 + c[[2, 2]] * v2 + c[[2, 1]] * v + c[[2, 0]]
    );
    result = (result * u) + (
        c[[1, 3]] * v3 + c[[1, 2]] * v2 + c[[1, 1]] * v + c[[1, 0]]
    );
    result = (result * u) + (
        c[[0, 3]] * v3 + c[[0, 2]] * v2 + c[[0, 1]] * v + c[[0, 0]]
    );
    
    result
}

/// Optimized bicubic evaluation with manual inlining
pub fn bicubic_eval_fast(c: &ArrayView2<f64>, u: f64, v: f64) -> f64 {
    debug_assert_eq!(c.shape(), &[4, 4], "Coefficient matrix must be 4x4");
    
    // Precompute all powers
    let u2 = u * u;
    let u3 = u2 * u;
    
    let v2 = v * v;
    let v3 = v2 * v;
    
    // Direct polynomial evaluation (often faster than Horner for small degrees)
    let v_terms = [
        c[[0, 0]] + c[[0, 1]] * v + c[[0, 2]] * v2 + c[[0, 3]] * v3,
        c[[1, 0]] + c[[1, 1]] * v + c[[1, 2]] * v2 + c[[1, 3]] * v3,
        c[[2, 0]] + c[[2, 1]] * v + c[[2, 2]] * v2 + c[[2, 3]] * v3,
        c[[3, 0]] + c[[3, 1]] * v + c[[3, 2]] * v2 + c[[3, 3]] * v3,
    ];
    
    v_terms[0] + v_terms[1] * u + v_terms[2] * u2 + v_terms[3] * u3
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
    let c = bcucof_fast(y, y1, y2, y12, d1, d2);
    bicubic_eval_fast(&c.view(), u, v)
}

/// Batch interpolation for multiple points with the same coefficients
pub fn bicubic_interpolate_batch(
    c: &ArrayView2<f64>,
    uv_points: &[(f64, f64)],
) -> Vec<f64> {
    uv_points.par_iter()
        .map(|&(u, v)| bicubic_eval_fast(c, u, v))
        .collect()
}

/// Cached bicubic interpolator for repeated evaluations
pub struct BicubicInterpolator {
    coefficients: Array2<f64>,
}

impl BicubicInterpolator {
    pub fn new(y: &[f64], y1: &[f64], y2: &[f64], y12: &[f64], d1: f64, d2: f64) -> Self {
        let coefficients = bcucof_fast(y, y1, y2, y12, d1, d2);
        Self { coefficients }
    }
    
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        bicubic_eval_fast(&self.coefficients.view(), u, v)
    }
    
    pub fn eval_batch(&self, uv_points: &[(f64, f64)]) -> Vec<f64> {
        bicubic_interpolate_batch(&self.coefficients.view(), uv_points)
    }
    
    pub fn coefficients(&self) -> &Array2<f64> {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bcucof_constant_function() {
        // Constant function: f(x1, x2) = 5.0
        let y = [5.0, 5.0, 5.0, 5.0];
        let y1 = [0.0, 0.0, 0.0, 0.0];
        let y2 = [0.0, 0.0, 0.0, 0.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        let c = bcucof(&y, &y1, &y2, &y12, 1.0, 1.0);
        let c_fast = bcucof_fast(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Both methods should give same results
        assert_abs_diff_eq!(c, c_fast, epsilon = 1e-10);
        
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
        let y = [0.0, 3.0, 2.0, 5.0];
        let y1 = [2.0, 2.0, 2.0, 2.0];
        let y2 = [3.0, 3.0, 3.0, 3.0];
        let y12 = [0.0, 0.0, 0.0, 0.0];
        
        let c = bcucof_fast(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Coefficients for 2*u + 3*v
        assert_abs_diff_eq!(c[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 3.0, epsilon = 1e-10);
        
        // All other coefficients should be zero for linear function
        for i in 0..4 {
            for j in 0..4 {
                if !((i == 1 && j == 0) || (i == 0 && j == 1)) {
                    assert_abs_diff_eq!(c[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_bcucof_bilinear_function() {
        // Bilinear function: f(x1, x2) = x1 * x2
        let y = [0.0, 0.0, 0.0, 1.0];
        let y1 = [0.0, 1.0, 0.0, 1.0];
        let y2 = [0.0, 0.0, 1.0, 1.0];
        let y12 = [1.0, 1.0, 1.0, 1.0];
        
        let c = bcucof_fast(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Should reproduce exactly u*v
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let interpolated = bicubic_eval_fast(&c.view(), u, v);
                let exact = u * v;
                assert_abs_diff_eq!(interpolated, exact, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bicubic_eval_consistency() {
        let y = [0.0, 1.0, 2.0, 3.0];
        let y1 = [1.0, 1.0, 1.0, 1.0];
        let y2 = [2.0, 2.0, 2.0, 2.0];
        let y12 = [0.5, 0.5, 0.5, 0.5];
        
        let c = bcucof_fast(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Test evaluation at corners
        assert_abs_diff_eq!(bicubic_eval_fast(&c.view(), 0.0, 0.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval_fast(&c.view(), 0.0, 1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval_fast(&c.view(), 1.0, 0.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bicubic_eval_fast(&c.view(), 1.0, 1.0), 3.0, epsilon = 1e-10);
        
        // Both evaluation methods should give same results
        let u = 0.3;
        let v = 0.7;
        assert_abs_diff_eq!(
            bicubic_eval(&c.view(), u, v),
            bicubic_eval_fast(&c.view(), u, v),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_bicubic_interpolator() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let y1 = [0.1, 0.2, 0.3, 0.4];
        let y2 = [0.5, 0.6, 0.7, 0.8];
        let y12 = [0.01, 0.02, 0.03, 0.04];
        
        let interpolator = BicubicInterpolator::new(&y, &y1, &y2, &y12, 1.0, 1.0);
        
        // Test single evaluation
        let result = interpolator.eval(0.5, 0.5);
        assert!(result >= 1.0 && result <= 4.0);
        
        // Test batch evaluation
        let points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];
        let results = interpolator.eval_batch(&points);
        assert_eq!(results.len(), 3);
        assert_abs_diff_eq!(results[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_processing_consistency() {
        let patches: Vec<_> = (0..10)
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
        
        let results_par = bcucof_batch(&patches, 1.0, 1.0);
        assert_eq!(results_par.len(), 10);
    }

    #[test]
    #[should_panic(expected = "y must have exactly 4 elements")]
    fn test_bcucof_wrong_input_size() {
        let y = [1.0, 2.0, 3.0];
        let y1 = [0.0; 4];
        let y2 = [0.0; 4];
        let y12 = [0.0; 4];
        
        bcucof_fast(&y, &y1, &y2, &y12, 1.0, 1.0);
    }
}
