use ndarray::{Array1, Array2};
use crate::constants;
use crate::NumrsError;

// Utility functions and common routines for numerical computations

/// Computes `sqrt(a^2 + b^2)` without destructive underflow or overflow.
pub fn pythag(a: f64, b: f64) -> f64 {
    let absa = a.abs();
    let absb = b.abs();
    if absa > absb {
        absa * (1.0 + (absb / absa).powi(2)).sqrt()
    } else if absb == 0.0 {
        0.0
    } else {
        absb * (1.0 + (absa / absb).powi(2)).sqrt()
    }
}

/// Returns `|a| * sgn(b)`
pub fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

/// Helper function to convert 1-based indexing to 0-based for matrix operations
pub fn get_matrix_row<'a>(matrix: &'a Array2<f64>, row: usize) -> Array1<f64> {
    matrix.row(row).to_owned()
}

/// Creates a vector with 0-based indexing (convenience wrapper)
pub fn create_vector(size: usize) -> Array1<f64> {
    Array1::zeros(size)
}

/// Frees vector (no-op in Rust due to ownership system, but provided for API compatibility)
pub fn free_vector(_vec: Array1<f64>) {
    // Automatic memory management - no explicit free needed
}

/// Check if value is approximately zero within machine epsilon
pub fn approx_zero(x: f64) -> bool {
    x.abs() < constants::EPSILON
}

/// Check if two values are approximately equal
pub fn approx_eq(a: f64, b: f64, tol: Option<f64>) -> bool {
    let tolerance = tol.unwrap_or(constants::SQRT_EPSILON);
    (a - b).abs() < tolerance
}

/// Sign function (returns sign of x)
pub fn signum(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

/// Swap two values
pub fn swap<T>(a: &mut T, b: &mut T) {
    std::mem::swap(a, b);
}

/// Maximum of two values
pub fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// Minimum of two values
pub fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

/// Square of a value
pub fn sqr(x: f64) -> f64 {
    x * x
}

/// Cube of a value
pub fn cube(x: f64) -> f64 {
    x * x * x
}

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Clamp value between min and max
pub fn clamp(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x < min_val {
        min_val
    } else if x > max_val {
        max_val
    } else {
        x
    }
}

/// Check if value is within range [min, max]
pub fn in_range(x: f64, min_val: f64, max_val: f64) -> bool {
    x >= min_val && x <= max_val
}

/// Convert degrees to radians
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees
pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}

/// Factorial function
pub fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

/// Double factorial function
pub fn double_factorial(n: i32) -> u64 {
    if n <= 0 {
        return 1;
    }
    
    let mut result = 1;
    let mut current = n;
    
    while current > 0 {
        result *= current as u64;
        current -= 2;
    }
    
    result
}

/// Binomial coefficient
pub fn binomial_coefficient(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    
    let k = min(k, n - k);
    let mut result = 1;
    
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    
    result
}

/// Check if number is even
pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}

/// Check if number is odd
pub fn is_odd(n: i32) -> bool {
    n % 2 != 0
}

/// Compute the next power of two
pub fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Check if number is power of two
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Vector dot product
pub fn dot_product(a: &[f64], b: &[f64]) -> Result<f64, NumrsError> {
    if a.len() != b.len() {
        return Err(NumrsError::DomainError("Vectors must have same length".to_string()));
    }
    
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

/// Vector norm (L2 norm)
pub fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Vector normalization
pub fn normalize_vector(v: &[f64]) -> Vec<f64> {
    let norm = vector_norm(v);
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Create an identity matrix of given size
pub fn identity_matrix(size: usize) -> Array2<f64> {
    let mut mat = Array2::zeros((size, size));
    for i in 0..size {
        mat[(i, i)] = 1.0;
    }
    mat
}

/// Matrix multiplication helper
pub fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, NumrsError> {
    if a.shape()[1] != b.shape()[0] {
        return Err(NumrsError::DomainError(
            "Matrix dimensions incompatible for multiplication".to_string()
        ));
    }
    
    let mut result = Array2::zeros((a.shape()[0], b.shape()[1]));
    
    for i in 0..a.shape()[0] {
        for j in 0..b.shape()[1] {
            for k in 0..a.shape()[1] {
                result[(i, j)] += a[(i, k)] * b[(k, j)];
            }
        }
    }
    
    Ok(result)
}

/// Matrix transpose
pub fn matrix_transpose(matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let mut result = Array2::zeros((cols, rows));
    
    for i in 0..rows {
        for j in 0..cols {
            result[(j, i)] = matrix[(i, j)];
        }
    }
    
    result
}

/// Print matrix for debugging
pub fn print_matrix(matrix: &Array2<f64>) {
    for i in 0..matrix.shape()[0] {
        for j in 0..matrix.shape()[1] {
            print!("{:8.4} ", matrix[(i, j)]);
        }
        println!();
    }
}

/// Print vector for debugging
pub fn print_vector(vector: &[f64]) {
    for &val in vector {
        print!("{:8.4} ", val);
    }
    println!();
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pythag() {
        assert_eq!(pythag(3.0, 4.0), 5.0);
        assert_eq!(pythag(-3.0, 4.0), 5.0);
        assert_eq!(pythag(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_sign() {
        assert_eq!(sign(5.0, -2.0), -5.0);
        assert_eq!(sign(5.0, 2.0), 5.0);
        assert_eq!(sign(5.0, 0.0), 5.0); // copysign(0.0) is positive
    }

    #[test]
    fn test_approx_eq() {
        assert!(approx_eq(1.0, 1.0 + 1e-9, None));
        assert!(!approx_eq(1.0, 1.1, None));
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 5), 1);
    }

    #[test]
    fn test_vector_operations() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 5.0, 6.0];
        
        assert_abs_diff_eq!(dot_product(&v1, &v2).unwrap(), 32.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vector_norm(&v1), 3.7416573867739413, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let result = matrix_multiply(&a, &b).unwrap();
        let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }
    }
}
