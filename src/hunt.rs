use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum HuntError {
    EmptyArray,
    UnsortedArray,
    InvalidGuess,
    OutOfRange,
}

impl fmt::Display for HuntError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HuntError::EmptyArray => write!(f, "Array cannot be empty"),
            HuntError::UnsortedArray => write!(f, "Array must be sorted"),
            HuntError::InvalidGuess => write!(f, "Initial guess is invalid"),
            HuntError::OutOfRange => write!(f, "Value out of array range"),
        }
    }
}

impl Error for HuntError {}

pub type HuntResult<T> = std::result::Result<T, HuntError>;

/// Hunts for the index j such that xx[j] ≤ x < xx[j+1] in a sorted array xx
/// Uses an initial guess and exponential search to quickly find the interval
pub fn hunt(xx: &[f32], x: f32, jlo_guess: usize) -> HuntResult<usize> {
    let n = xx.len();
    
    // Input validation
    if n == 0 {
        return Err(HuntError::EmptyArray);
    }
    
    // Check if array is sorted
    let ascending = if n > 1 {
        let is_ascending = xx[n - 1] >= xx[0];
        for i in 1..n {
            if is_ascending && xx[i] < xx[i - 1] {
                return Err(HuntError::UnsortedArray);
            } else if !is_ascending && xx[i] > xx[i - 1] {
                return Err(HuntError::UnsortedArray);
            }
        }
        is_ascending
    } else {
        true // Single element is always "sorted"
    };

    // Handle edge cases
    if ascending {
        if x < xx[0] {
            return Err(HuntError::OutOfRange);
        }
        if x > xx[n - 1] {
            return Err(HuntError::OutOfRange);
        }
        if (x - xx[n - 1]).abs() < f32::EPSILON {
            return Ok(n - 2); // x equals last element
        }
    } else {
        if x > xx[0] {
            return Err(HuntError::OutOfRange);
        }
        if x < xx[n - 1] {
            return Err(HuntError::OutOfRange);
        }
        if (x - xx[n - 1]).abs() < f32::EPSILON {
            return Ok(n - 2); // x equals last element
        }
    }

    let mut jlo = if jlo_guess < n {
        jlo_guess
    } else {
        0
    };

    let mut jhi;

    // If guess is invalid or out of bounds, use binary search
    if jlo == 0 || jlo >= n {
        jlo = 0;
        jhi = n;
    } else {
        // Check if the guess is correct
        if (ascending && xx[jlo] <= x && x < xx[jlo + 1])
            || (!ascending && xx[jlo] >= x && x > xx[jlo + 1])
        {
            return Ok(jlo);
        }

        // Determine search direction and perform exponential search
        if (x >= xx[jlo]) == ascending {
            // Search upward
            let mut inc = 1;
            jhi = jlo + 1;
            
            while jhi < n && (x >= xx[jhi]) == ascending {
                jlo = jhi;
                inc *= 2;
                jhi = jlo + inc;
                if jhi > n {
                    jhi = n;
                    break;
                }
            }
        } else {
            // Search downward
            let mut inc = 1;
            jhi = jlo;
            
            while jlo > 0 && (x < xx[jlo]) == ascending {
                jhi = jlo;
                inc *= 2;
                if inc > jhi {
                    jlo = 0;
                    break;
                } else {
                    jlo = jhi - inc;
                }
            }
        }
    }

    // Final binary search within the bracketed interval
    while jhi - jlo > 1 {
        let jm = (jhi + jlo) / 2;
        
        if (x >= xx[jm]) == ascending {
            jlo = jm;
        } else {
            jhi = jm;
        }
    }

    // Handle exact matches at boundaries
    if (x - xx[0]).abs() < f32::EPSILON {
        Ok(0)
    } else if (x - xx[n - 1]).abs() < f32::EPSILON {
        Ok(n - 2)
    } else {
        Ok(jlo)
    }
}

/// Simple binary search for comparison
pub fn hunt_binary_search(xx: &[f32], x: f32, _jlo_guess: usize) -> HuntResult<usize> {
    let n = xx.len();
    
    if n == 0 {
        return Err(HuntError::EmptyArray);
    }
    
    // Check if array is sorted in ascending order
    for i in 1..n {
        if xx[i] < xx[i - 1] {
            return Err(HuntError::UnsortedArray);
        }
    }
    
    if x < xx[0] || x > xx[n - 1] {
        return Err(HuntError::OutOfRange);
    }
    if (x - xx[n - 1]).abs() < f32::EPSILON {
        return Ok(n - 2);
    }

    let mut left = 0;
    let mut right = n - 1;
    
    while left <= right {
        let mid = (left + right) / 2;
        
        if x < xx[mid] {
            right = mid - 1;
        } else if x >= xx[mid + 1] {
            left = mid + 1;
        } else {
            return Ok(mid);
        }
    }
    
    Ok(left)
}

/// Creates test arrays for various scenarios
pub fn create_test_array_ascending(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

pub fn create_test_array_descending(n: usize) -> Vec<f32> {
    (0..n).rev().map(|i| i as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hunt_ascending_with_good_guess() {
        let xx = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Good guess near the actual position
        assert_eq!(hunt(&xx, 2.5, 2).unwrap(), 2);
        assert_eq!(hunt(&xx, 3.5, 3).unwrap(), 3);
        assert_eq!(hunt(&xx, 1.5, 1).unwrap(), 1);
    }

    #[test]
    fn test_hunt_ascending_with_bad_guess() {
        let xx = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Bad guess - should still find correct position
        assert_eq!(hunt(&xx, 2.5, 0).unwrap(), 2);
        assert_eq!(hunt(&xx, 3.5, 5).unwrap(), 3);
        assert_eq!(hunt(&xx, 1.5, 4).unwrap(), 1);
    }

    #[test]
    fn test_hunt_descending_with_guess() {
        let xx = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        
        assert_eq!(hunt(&xx, 3.5, 2).unwrap(), 1);
        assert_eq!(hunt(&xx, 2.5, 3).unwrap(), 2);
        assert_eq!(hunt(&xx, 1.5, 4).unwrap(), 3);
    }

    #[test]
    fn test_hunt_exact_matches() {
        let xx = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        
        // Exact matches should return the left index
        assert_eq!(hunt(&xx, 1.0, 1).unwrap(), 1);
        assert_eq!(hunt(&xx, 2.0, 2).unwrap(), 2);
        assert_eq!(hunt(&xx, 3.0, 3).unwrap(), 3);
        assert_eq!(hunt(&xx, 4.0, 4).unwrap(), 3); // Special case for last element
    }

    #[test]
    fn test_hunt_single_element() {
        let xx = vec![5.0];
        
        // For single element, x must equal that element
        assert_eq!(hunt(&xx, 5.0, 0).unwrap(), 0);
        
        let result = hunt(&xx, 4.0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HuntError::OutOfRange);
    }

    #[test]
    fn test_hunt_empty_array() {
        let xx: Vec<f32> = vec![];
        
        let result = hunt(&xx, 1.0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HuntError::EmptyArray);
    }

    #[test]
    fn test_hunt_unsorted_array() {
        let xx = vec![1.0, 3.0, 2.0, 4.0]; // Not sorted
        
        let result = hunt(&xx, 2.5, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HuntError::UnsortedArray);
    }

    #[test]
    fn test_hunt_out_of_range() {
        let xx = vec![1.0, 2.0, 3.0, 4.0];
        
        // Below range
        let result = hunt(&xx, 0.5, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HuntError::OutOfRange);
        
        // Above range
        let result = hunt(&xx, 4.5, 2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HuntError::OutOfRange);
    }

    #[test]
    fn test_hunt_large_array_with_guess() {
        let n = 1000;
        let xx = create_test_array_ascending(n);
        
        // Test with good guesses
        assert_eq!(hunt(&xx, 250.5, 250).unwrap(), 250);
        assert_eq!(hunt(&xx, 500.0, 500).unwrap(), 500);
        assert_eq!(hunt(&xx, 750.25, 750).unwrap(), 750);
        
        // Test with bad guesses
        assert_eq!(hunt(&xx, 250.5, 0).unwrap(), 250);
        assert_eq!(hunt(&xx, 500.0, 999).unwrap(), 500);
    }

    #[test]
    fn test_hunt_exponential_search_behavior() {
        let xx = create_test_array_ascending(100);
        
        // Test cases where exponential search should be faster than binary search
        // Start with a guess that's far from the target
        assert_eq!(hunt(&xx, 75.5, 10).unwrap(), 75); // Far guess, should use exponential search
        assert_eq!(hunt(&xx, 25.5, 90).unwrap(), 25); // Far guess in opposite direction
    }

    #[test]
    fn test_hunt_vs_binary_search() {
        let xx = create_test_array_ascending(50);
        
        for x in (0..100).map(|i| i as f32 * 0.5) {
            if x < xx[0] || x >= xx[49] {
                continue; // Skip out-of-range values
            }
            
            // Test with various guesses
            for &guess in &[0, 10, 25, 40, 49] {
                let result1 = hunt(&xx, x, guess).unwrap();
                let result2 = hunt_binary_search(&xx, x, guess).unwrap();
                
                // Both methods should give the same result
                assert_eq!(result1, result2, "Mismatch at x = {} with guess = {}", x, guess);
            }
        }
    }

    #[test]
    fn test_hunt_guess_at_boundaries() {
        let xx = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Guess at lower boundary
        assert_eq!(hunt(&xx, 3.5, 0).unwrap(), 3);
        
        // Guess at upper boundary
        assert_eq!(hunt(&xx, 2.5, 4).unwrap(), 2);
    }

    #[test]
    fn test_hunt_duplicate_values() {
        let xx = vec![1.0, 2.0, 2.0, 3.0, 4.0]; // Non-decreasing
        
        assert_eq!(hunt(&xx, 2.0, 1).unwrap(), 2); // Returns last index with value <= x
        assert_eq!(hunt(&xx, 2.5, 2).unwrap(), 2);
    }

    #[test]
    fn test_hunt_random_arrays_with_guesses() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..10 {
            let n = rng.gen_range(20..100);
            let ascending = rng.gen_bool(0.5);
            
            let mut xx: Vec<f32> = if ascending {
                (0..n).map(|i| i as f32).collect()
            } else {
                (0..n).rev().map(|i| i as f32).collect()
            };
            
            // Ensure the array is properly sorted
            if ascending {
                xx.sort_by(|a, b| a.partial_cmp(b).unwrap());
            } else {
                xx.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }
            
            // Test random values with random guesses
            for _ in 0..5 {
                let x_min = xx[0];
                let x_max = xx[n - 1];
                let x = if ascending {
                    rng.gen_range(x_min..x_max)
                } else {
                    rng.gen_range(x_max..x_min)
                };
                
                let guess = rng.gen_range(0..n);
                let result = hunt(&xx, x, guess).unwrap();
                
                // Verify the result satisfies the bracket condition
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
    fn test_hunt_performance_advantage() {
        // This test demonstrates the performance advantage of hunt over pure binary search
        // when good guesses are available
        let xx = create_test_array_ascending(1000);
        
        // Simulate a scenario where we're searching sequentially with good guesses
        let mut previous_guess = 0;
        
        for x in (100..900).step_by(10).map(|i| i as f32 + 0.5) {
            let result = hunt(&xx, x, previous_guess).unwrap();
            previous_guess = result;
            
            // Verify correctness
            assert!(xx[result] <= x);
            assert!(x < xx[result + 1]);
        }
    }

    #[test]
    fn test_hunt_edge_cases() {
        let xx = vec![1.0, 2.0, 3.0, 4.0];
        
        // Edge case: guess is exactly at array length
        assert_eq!(hunt(&xx, 2.5, 4).unwrap(), 2);
        
        // Edge case: guess is beyond array length
        assert_eq!(hunt(&xx, 2.5, 10).unwrap(), 2);
        
        // Edge case: guess is 0
        assert_eq!(hunt(&xx, 2.5, 0).unwrap(), 2);
    }
}
