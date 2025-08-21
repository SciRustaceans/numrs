use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum LocateError {
    EmptyArray,
    UnsortedArray,
    OutOfRange,
}

impl fmt::Display for LocateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LocateError::EmptyArray => write!(f, "Array cannot be empty"),
            LocateError::UnsortedArray => write!(f, "Array must be sorted"),
            LocateError::OutOfRange => write!(f, "Value out of array range"),
        }
    }
}

impl Error for LocateError {}

pub type LocateResult<T> = std::result::Result<T, LocateError>;

/// Finds the index j such that xx[j] ≤ x < xx[j+1] for a sorted array xx
/// Returns the index j (0-based) for the left bracket of the interval containing x
pub fn locate(xx: &[f32], x: f32) -> LocateResult<usize> {
    let n = xx.len();
    
    // Input validation
    if n == 0 {
        return Err(LocateError::EmptyArray);
    }
    
    // Check if array is sorted
    let ascending = if n > 1 {
        let is_ascending = xx[n - 1] >= xx[0];
        for i in 1..n {
            if is_ascending && xx[i] < xx[i - 1] {
                return Err(LocateError::UnsortedArray);
            } else if !is_ascending && xx[i] > xx[i - 1] {
                return Err(LocateError::UnsortedArray);
            }
        }
        is_ascending
    } else {
        true // Single element is always "sorted"
    };

    // Handle edge cases
    if ascending {
        if x < xx[0] {
            return Err(LocateError::OutOfRange);
        }
        if x >= xx[n - 1] {
            return if (x - xx[n - 1]).abs() < f32::EPSILON {
                Ok(n - 2) // x equals last element, return n-2 for 0-based
            } else {
                Err(LocateError::OutOfRange)
            };
        }
    } else {
        if x > xx[0] {
            return Err(LocateError::OutOfRange);
        }
        if x <= xx[n - 1] {
            return if (x - xx[n - 1]).abs() < f32::EPSILON {
                Ok(n - 2) // x equals last element, return n-2 for 0-based
            } else {
                Err(LocateError::OutOfRange)
            };
        }
    }

    // Binary search
    let mut jl = 0;
    let mut ju = n;

    while ju - jl > 1 {
        let jm = (ju + jl) / 2;
        
        if (x >= xx[jm]) == ascending {
            jl = jm;
        } else {
            ju = jm;
        }
    }

    // Handle exact matches at boundaries (0-based adjustment)
    if (x - xx[0]).abs() < f32::EPSILON {
        Ok(0)
    } else if (x - xx[n - 1]).abs() < f32::EPSILON {
        Ok(n - 2)
    } else {
        Ok(jl)
    }
}

/// Alternative implementation using binary search for comparison
pub fn locate_binary_search(xx: &[f32], x: f32) -> LocateResult<usize> {
    let n = xx.len();
    
    if n == 0 {
        return Err(LocateError::EmptyArray);
    }
    
    // Check if array is sorted in ascending order
    for i in 1..n {
        if xx[i] < xx[i - 1] {
            return Err(LocateError::UnsortedArray);
        }
    }
    
    if x < xx[0] {
        return Err(LocateError::OutOfRange);
    }
    if x >= xx[n - 1] {
        return if (x - xx[n - 1]).abs() < f32::EPSILON {
            Ok(n - 2)
        } else {
            Err(LocateError::OutOfRange)
        };
    }

    let mut left = 0;
    let mut right = n - 1;
    
    while left <= right {
        let mid = (left + right) / 2;
        
        match xx[mid].partial_cmp(&x).unwrap() {
            Ordering::Less => left = mid + 1,
            Ordering::Greater => right = mid - 1,
            Ordering::Equal => return Ok(mid),
        }
    }
    
    Ok(right)
}

/// Creates test arrays for various scenarios
pub fn create_test_array_ascending(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

pub fn create_test_array_descending(n: usize) -> Vec<f32> {
    (0..n).rev().map(|i| i as f32).collect()
}

pub fn create_test_array_float(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.5).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_locate_ascending_integer() {
        let xx = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test various positions
        assert_eq!(locate(&xx, 0.5).unwrap(), 0);
        assert_eq!(locate(&xx, 1.5).unwrap(), 1);
        assert_eq!(locate(&xx, 2.5).unwrap(), 2);
        assert_eq!(locate(&xx, 3.5).unwrap(), 3);
        assert_eq!(locate(&xx, 4.5).unwrap(), 4);
    }

    #[test]
    fn test_locate_ascending_float() {
        let xx = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        
        assert_eq!(locate(&xx, 0.25).unwrap(), 0);
        assert_eq!(locate(&xx, 0.75).unwrap(), 1);
        assert_eq!(locate(&xx, 1.25).unwrap(), 2);
        assert_eq!(locate(&xx, 1.75).unwrap(), 3);
        assert_eq!(locate(&xx, 2.25).unwrap(), 4);
    }

    #[test]
    fn test_locate_descending() {
        let xx = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        
        assert_eq!(locate(&xx, 4.5).unwrap(), 0);
        assert_eq!(locate(&xx, 3.5).unwrap(), 1);
        assert_eq!(locate(&xx, 2.5).unwrap(), 2);
        assert_eq!(locate(&xx, 1.5).unwrap(), 3);
        assert_eq!(locate(&xx, 0.5).unwrap(), 4);
    }

    #[test]
    fn test_locate_exact_matches() {
        let xx = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        
        // Exact matches should return the left index
        assert_eq!(locate(&xx, 0.0).unwrap(), 0);
        assert_eq!(locate(&xx, 1.0).unwrap(), 1);
        assert_eq!(locate(&xx, 2.0).unwrap(), 2);
        assert_eq!(locate(&xx, 3.0).unwrap(), 3);
        assert_eq!(locate(&xx, 4.0).unwrap(), 3); // Special case for last element
    }

    #[test]
    fn test_locate_single_element() {
        let xx = vec![5.0];
        
        // For single element, x must equal that element
        assert_eq!(locate(&xx, 5.0).unwrap(), 0);
        
        let result = locate(&xx, 4.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::OutOfRange);
        
        let result = locate(&xx, 6.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::OutOfRange);
    }

    #[test]
    fn test_locate_empty_array() {
        let xx: Vec<f32> = vec![];
        
        let result = locate(&xx, 1.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::EmptyArray);
    }

    #[test]
    fn test_locate_unsorted_array() {
        let xx = vec![1.0, 3.0, 2.0, 4.0]; // Not sorted
        
        let result = locate(&xx, 2.5);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::UnsortedArray);
    }

    #[test]
    fn test_locate_out_of_range() {
        let xx = vec![1.0, 2.0, 3.0, 4.0];
        
        // Below range
        let result = locate(&xx, 0.5);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::OutOfRange);
        
        // Above range
        let result = locate(&xx, 4.5);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LocateError::OutOfRange);
    }

    #[test]
    fn test_locate_boundary_cases() {
        let xx = vec![1.0, 2.0, 3.0, 4.0];
        
        // Just above lower boundary
        assert_eq!(locate(&xx, 1.0 + f32::EPSILON).unwrap(), 0);
        
        // Just below upper boundary
        assert_eq!(locate(&xx, 4.0 - f32::EPSILON).unwrap(), 2);
    }

    #[test]
    fn test_locate_large_array() {
        let n = 1000;
        let xx = create_test_array_ascending(n);
        
        // Test various positions in large array
        assert_eq!(locate(&xx, 250.5).unwrap(), 250);
        assert_eq!(locate(&xx, 500.0).unwrap(), 500);
        assert_eq!(locate(&xx, 750.25).unwrap(), 750);
        assert_eq!(locate(&xx, 999.0).unwrap(), 998); // Last element special case
    }

    #[test]
    fn test_locate_duplicate_values() {
        // Arrays with duplicate values should still be considered sorted
        // if they are non-decreasing/non-increasing
        let xx_asc = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let xx_desc = vec![4.0, 3.0, 3.0, 2.0, 1.0];
        
        assert_eq!(locate(&xx_asc, 2.0).unwrap(), 2); // Returns last index with value <= x
        assert_eq!(locate(&xx_asc, 2.5).unwrap(), 2);
        
        assert_eq!(locate(&xx_desc, 3.0).unwrap(), 2); // Returns last index with value >= x
        assert_eq!(locate(&xx_desc, 2.5).unwrap(), 3);
    }

    #[test]
    fn test_locate_vs_binary_search() {
        let xx = create_test_array_float(20);
        
        for x in (0..40).map(|i| i as f32 * 0.25) {
            if x < xx[0] || x >= xx[19] {
                continue; // Skip out-of-range values
            }
            
            let result1 = locate(&xx, x).unwrap();
            let result2 = locate_binary_search(&xx, x).unwrap();
            
            // Both methods should give the same result for ascending arrays
            assert_eq!(result1, result2, "Mismatch at x = {}", x);
        }
    }

    #[test]
    fn test_locate_precision() {
        let xx = vec![1.0, 2.0, 3.0, 4.0];
        
        // Test with values very close to boundaries
        assert_eq!(locate(&xx, 1.0 + 1e-10).unwrap(), 0);
        assert_eq!(locate(&xx, 2.0 - 1e-10).unwrap(), 0);
        assert_eq!(locate(&xx, 2.0 + 1e-10).unwrap(), 1);
        assert_eq!(locate(&xx, 3.0 - 1e-10).unwrap(), 1);
    }

    #[test]
    fn test_locate_random_arrays() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..10 {
            let n = rng.gen_range(10..50);
            let ascending = rng.gen_bool(0.5);
            
            let mut xx: Vec<f32> = if ascending {
                (0..n).map(|i| i as f32 + rng.gen_range(-0.1..0.1)).collect()
            } else {
                (0..n).rev().map(|i| i as f32 + rng.gen_range(-0.1..0.1)).collect()
            };
            
            // Ensure the array is properly sorted
            if ascending {
                xx.sort_by(|a, b| a.partial_cmp(b).unwrap());
            } else {
                xx.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }
            
            // Test random values within range
            for _ in 0..5 {
                let x_min = xx[0];
                let x_max = xx[n - 1];
                let x = if ascending {
                    rng.gen_range(x_min..x_max)
                } else {
                    rng.gen_range(x_max..x_min)
                };
                
                let result = locate(&xx, x).unwrap();
                
                // Verify the result satisfies: xx[result] ≤ x < xx[result+1] for ascending
                // or xx[result] ≥ x > xx[result+1] for descending
                if ascending {
                    assert!(xx[result] <= x, "xx[{}] = {} > x = {}", result, xx[result], x);
                    assert!(x < xx[result + 1], "x = {} >= xx[{}] = {}", x, result + 1, xx[result + 1]);
                } else {
                    assert!(xx[result] >= x, "xx[{}] = {} < x = {}", result, xx[result], x);
                    assert!(x > xx[result + 1], "x = {} <= xx[{}] = {}", x, result + 1, xx[result + 1]);
                }
            }
        }
    }

    #[test]
    fn test_locate_edge_case_floats() {
        // Test with floating point edge cases
        let xx = vec![f32::MIN, -1.0, 0.0, 1.0, f32::MAX];
        
        assert_eq!(locate(&xx, -0.5).unwrap(), 1);
        assert_eq!(locate(&xx, 0.5).unwrap(), 2);
        
        // Test near boundaries
        assert_eq!(locate(&xx, f32::MIN + f32::EPSILON).unwrap(), 0);
        assert_eq!(locate(&xx, f32::MAX - f32::EPSILON).unwrap(), 3);
    }
}
