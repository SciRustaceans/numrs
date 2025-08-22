use ndarray::{Array1, Array2};
//! Helper functions for numerical calculations.

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
// Unit tests are placed in the same file as the code they are testing.
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

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
}
