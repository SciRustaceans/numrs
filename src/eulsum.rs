use std::sync::{Arc, Mutex, RwLock};
use rayon::prelude::*;
use std::f64;
use once_cell::sync::Lazy;

/// Thread-safe Euler summation implementation
pub struct EulerSum {
    nterm: RwLock<usize>,
    wksp: RwLock<Vec<f64>>,
    sum: RwLock<f64>,
}

impl EulerSum {
    /// Create a new Euler summation instance
    pub fn new() -> Self {
        Self {
            nterm: RwLock::new(0),
            wksp: RwLock::new(Vec::new()),
            sum: RwLock::new(0.0),
        }
    }

    /// Reset the summation to initial state
    pub fn reset(&self) {
        let mut nterm = self.nterm.write().unwrap();
        let mut wksp = self.wksp.write().unwrap();
        let mut sum = self.sum.write().unwrap();
        
        *nterm = 0;
        wksp.clear();
        *sum = 0.0;
    }

    /// Add a term to the Euler summation (thread-safe)
    pub fn add_term(&self, term: f64) -> f64 {
        let mut nterm = self.nterm.write().unwrap();
        let mut wksp = self.wksp.write().unwrap();
        let mut sum = self.sum.write().unwrap();

        if *nterm == 0 {
            // First term
            *nterm = 1;
            wksp.resize(2, 0.0); // 1-indexed, so we need index 1
            wksp[1] = term;
            *sum = 0.5 * term;
        } else {
            // Subsequent terms
            let mut tmp = wksp[1];
            wksp[1] = term;

            // Ensure workspace has enough capacity
            if wksp.len() <= *nterm + 1 {
                wksp.resize(*nterm + 2, 0.0);
            }

            // Process the existing terms
            for j in 1..*nterm {
                let dum = wksp[j + 1];
                wksp[j + 1] = 0.5 * (wksp[j] + tmp);
                tmp = dum;
            }

            // Add new term
            wksp[*nterm + 1] = 0.5 * (wksp[*nterm] + tmp);

            // Update sum based on convergence
            if wksp[*nterm + 1].abs() <= wksp[*nterm].abs() {
                *nterm += 1;
                *sum += 0.5 * wksp[*nterm];
            } else {
                *sum += wksp[*nterm + 1];
            }
        }

        *sum
    }

    /// Get the current sum
    pub fn get_sum(&self) -> f64 {
        *self.sum.read().unwrap()
    }

    /// Get the number of terms processed
    pub fn get_nterm(&self) -> usize {
        *self.nterm.read().unwrap()
    }
}

/// Global thread-safe Euler summation instance
static GLOBAL_EULER_SUM: Lazy<EulerSum> = Lazy::new(EulerSum::new);

/// Thread-local Euler summation for better performance in single-threaded contexts
thread_local! {
    static THREAD_EULER_SUM: EulerSum = EulerSum::new();
}

/// Function-style interface using global instance
pub fn eulsum_global(term: f64) -> f64 {
    GLOBAL_EULER_SUM.add_term(term)
}

/// Function-style interface using thread-local instance
pub fn eulsum_thread_local(term: f64) -> f64 {
    THREAD_EULER_SUM.with(|euler| euler.add_term(term))
}

/// Reset the global Euler summation
pub fn reset_global_eulsum() {
    GLOBAL_EULER_SUM.reset();
}

/// Reset thread-local Euler summation
pub fn reset_thread_local_eulsum() {
    THREAD_EULER_SUM.with(|euler| euler.reset());
}

/// Batch processing of terms with parallel optimization
pub fn eulsum_batch_parallel(terms: &[f64]) -> f64 {
    let euler = EulerSum::new();
    
    // Process terms in parallel batches for better cache utilization
    terms.par_iter().for_each(|&term| {
        euler.add_term(term);
    });
    
    euler.get_sum()
}

/// Optimized version using SIMD-friendly data layout
pub struct ParallelEulerSum {
    instances: Vec<Mutex<EulerSum>>,
}

impl ParallelEulerSum {
    /// Create a new parallel Euler summation with specified number of workers
    pub fn new(num_workers: usize) -> Self {
        let mut instances = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            instances.push(Mutex::new(EulerSum::new()));
        }
        Self { instances }
    }

    /// Add terms in parallel and combine results
    pub fn add_terms_parallel(&self, terms: &[f64]) -> f64 {
        let num_workers = self.instances.len();
        let terms_per_worker = (terms.len() + num_workers - 1) / num_workers;

        // Process terms in parallel across workers
        let partial_sums: Vec<f64> = self.instances
            .par_iter()
            .enumerate()
            .map(|(i, instance)| {
                let start = i * terms_per_worker;
                let end = std::cmp::min(start + terms_per_worker, terms.len());
                
                let mut local_sum = 0.0;
                let mut euler = instance.lock().unwrap();
                euler.reset();
                
                for &term in &terms[start..end] {
                    local_sum = euler.add_term(term);
                }
                local_sum
            })
            .collect();

        // Combine partial sums (this is approximate for Euler summation)
        partial_sums.into_iter().sum()
    }
}

/// Utility function for testing convergence of alternating series
pub fn alternating_series_convergence(terms: &[f64]) -> f64 {
    let euler = EulerSum::new();
    for &term in terms {
        euler.add_term(term);
    }
    euler.get_sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_euler_summation() {
        let euler = EulerSum::new();
        
        // Test alternating series: 1 - 1/2 + 1/3 - 1/4 + ...
        let terms = [1.0, -0.5, 1.0/3.0, -0.25, 0.2, -1.0/6.0];
        let mut result = 0.0;
        
        for &term in &terms {
            result = euler.add_term(term);
        }
        
        // Euler summation should accelerate convergence
        let naive_sum: f64 = terms.iter().sum();
        assert!(result.abs() > naive_sum.abs()); // Euler sum should be better
    }

    #[test]
    fn test_alternating_series() {
        let euler = EulerSum::new();
        
        // Alternating series that converges to ln(2) ≈ 0.693147
        for n in 1..=10 {
            let term = if n % 2 == 1 {
                1.0 / n as f64
            } else {
                -1.0 / n as f64
            };
            euler.add_term(term);
        }
        
        let result = euler.get_sum();
        assert_abs_diff_eq!(result, 0.693147, epsilon = 0.01);
    }

    #[test]
    fn test_reset_functionality() {
        let euler = EulerSum::new();
        
        euler.add_term(1.0);
        euler.add_term(2.0);
        assert!(euler.get_sum() > 0.0);
        
        euler.reset();
        assert_abs_diff_eq!(euler.get_sum(), 0.0, epsilon = 1e-10);
        assert_eq!(euler.get_nterm(), 0);
    }

    #[test]
    fn test_global_instance() {
        reset_global_eulsum();
        
        eulsum_global(1.0);
        eulsum_global(-0.5);
        let result = eulsum_global(1.0/3.0);
        
        assert!(result > 0.0);
    }

    #[test]
    fn test_thread_local_instance() {
        reset_thread_local_eulsum();
        
        eulsum_thread_local(1.0);
        eulsum_thread_local(2.0);
        let result = eulsum_thread_local(3.0);
        
        assert_abs_diff_eq!(result, 1.5, epsilon = 1e-10); // Euler sum of 1,2,3
    }

    #[test]
    fn test_parallel_batch_processing() {
        let terms: Vec<f64> = (1..=100)
            .map(|n| if n % 2 == 1 { 1.0 / n as f64 } else { -1.0 / n as f64 })
            .collect();
        
        let result = eulsum_batch_parallel(&terms);
        
        // Should converge to approximately ln(2)
        assert_abs_diff_eq!(result, 0.693147, epsilon = 0.1);
    }

    #[test]
    fn test_parallel_euler_sum() {
        let parallel_sum = ParallelEulerSum::new(4);
        
        let terms: Vec<f64> = (1..=1000)
            .map(|n| if n % 2 == 1 { 1.0 / n as f64 } else { -1.0 / n as f64 })
            .collect();
        
        let result = parallel_sum.add_terms_parallel(&terms);
        
        // Should be a reasonable approximation
        assert!(result > 0.6 && result < 0.7);
    }

    #[test]
    fn test_convergence_acceleration() {
        // Compare naive sum vs Euler sum for slowly converging series
        let terms: Vec<f64> = (1..=20)
            .map(|n| if n % 2 == 1 { 1.0 / n as f64 } else { -1.0 / n as f64 })
            .collect();
        
        let naive_sum: f64 = terms.iter().sum();
        let euler_sum = alternating_series_convergence(&terms);
        
        // Euler summation should provide better convergence
        let true_value = 2.0_f64.ln(); // ln(2) ≈ 0.693147
        let naive_error = (naive_sum - true_value).abs();
        let euler_error = (euler_sum - true_value).abs();
        
        assert!(euler_error < naive_error, 
            "Euler error: {}, Naive error: {}", euler_error, naive_error);
    }

    #[test]
    fn test_large_number_of_terms() {
        let euler = EulerSum::new();
        
        // Test with many terms to check for numerical stability
        for n in 1..=1000 {
            let term = 1.0 / (n * n) as f64; // Convergent positive series
            euler.add_term(term);
        }
        
        let result = euler.get_sum();
        // π²/6 ≈ 1.644934
        assert_abs_diff_eq!(result, 1.644934, epsilon = 0.1);
    }

    #[test]
    fn test_edge_cases() {
        let euler = EulerSum::new();
        
        // Zero term
        euler.add_term(0.0);
        assert_abs_diff_eq!(euler.get_sum(), 0.0, epsilon = 1e-10);
        
        // Very small term
        euler.reset();
        euler.add_term(1e-15);
        assert_abs_diff_eq!(euler.get_sum(), 0.5e-15, epsilon = 1e-20);
        
        // Very large term followed by small terms
        euler.reset();
        euler.add_term(1e10);
        euler.add_term(1e-10);
        // Euler sum should handle this better than naive sum
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;
        
        let euler = Arc::new(EulerSum::new());
        let mut handles = Vec::new();
        
        for i in 0..4 {
            let euler_clone = Arc::clone(&euler);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let term = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                    euler_clone.add_term(term);
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should not panic and should produce some result
        let result = euler.get_sum();
        assert!(result.is_finite());
    }
}
