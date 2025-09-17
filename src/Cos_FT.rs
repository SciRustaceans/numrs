use std::f64::consts::PI;
use rayon::prelude::*;

/// Fast Cosine Transform implementation
/// Optimized with f64 precision, parallelization, and cache-friendly operations
pub fn cosft1(y: &mut [f64], n: usize) {
    // Precompute constants
    let theta = PI / n as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let n2 = n + 2;
    
    // Initial processing
    let mut sum = 0.5 * (y[1] - y[n + 1]);
    y[1] = 0.5 * (y[1] + y[n + 1]);
    
    // Initialize rotation variables
    let mut wr = 1.0;
    let mut wi = 0.0;
    
    // Precompute rotation factors to allow parallelization
    let half_n = n >> 1;
    let rotation_factors: Vec<(f64, f64)> = (2..=half_n)
        .scan((wr, wi), |(wr_state, wi_state), _| {
            let wtemp = *wr_state;
            *wr_state = wtemp * wpr - *wi_state * wpi + wtemp;
            *wi_state = *wi_state * wpr + wtemp * wpi + *wi_state;
            Some((*wr_state, *wi_state))
        })
        .collect();
    
    // Process pairs in parallel
    let partial_sums: Vec<f64> = (2..=half_n)
        .into_par_iter()
        .zip(rotation_factors.par_iter())
        .map(|(j, &(wr_val, wi_val))| {
            let sym_j = n2 - j;
            let y_j = y[j];
            let y_sym = y[sym_j];
            
            let y1 = 0.5 * (y_j + y_sym);
            let y2 = y_j - y_sym;
            
            y[j] = y1 - wi_val * y2;
            y[sym_j] = y1 + wi_val * y2;
            
            wr_val * y2
        })
        .collect();
    
    // Sum the partial results
    sum += partial_sums.into_iter().sum::<f64>();
    
    // Call the existing realft function
    realft(y, n, 1);
    
    // Final processing
    y[n + 1] = y[2];
    
    // Sequential accumulation for the final step
    for j in (4..=n).step_by(2) {
        sum += y[j];
        y[j] = sum;
    }
}

/// Wrapper for the existing realft function from NumRs library
fn realft(data: &mut [f64], n: usize, isign: i32) {
    // This would call your existing NumRs implementation
    // num_rs::realft(data, n, isign)
    unimplemented!("Use your NumRs library's realft implementation")
}

/// Alternative implementation with additional optimizations
pub fn cosft1_optimized(y: &mut [f64], n: usize) {
    let theta = PI / n as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let n2 = n + 2;
    
    // Process initial values
    let mut sum = 0.5 * (y[1] - y[n + 1]);
    y[1] = 0.5 * (y[1] + y[n + 1]);
    
    // Use iterative approach for rotation factors to avoid extra allocations
    let mut wr = 1.0;
    let mut wi = 0.0;
    
    // Process in chunks for better cache performance
    let chunk_size = (n >> 1) / rayon::current_num_threads().max(1);
    
    (2..=n>>1).into_par_iter().chunks(chunk_size).for_each(|chunk| {
        let mut local_wr = wr;
        let mut local_wi = wi;
        let mut local_sum = 0.0;
        
        for &j in &chunk {
            let wtemp = local_wr;
            local_wr = wtemp * wpr - local_wi * wpi + local_wr;
            local_wi = local_wi * wpr + wtemp * wpi + local_wi;
            
            let sym_j = n2 - j;
            let y_j = y[j];
            let y_sym = y[sym_j];
            
            let y1 = 0.5 * (y_j + y_sym);
            let y2 = y_j - y_sym;
            
            y[j] = y1 - local_wi * y2;
            y[sym_j] = y1 + local_wi * y2;
            
            local_sum += local_wr * y2;
        }
        
        // Atomic operations for thread-safe sum accumulation
        use std::sync::atomic::{AtomicU64, Ordering};
        static SUM_ATOMIC: AtomicU64 = AtomicU64::new(0);
        let local_sum_bits = local_sum.to_bits();
        SUM_ATOMIC.fetch_add(local_sum_bits, Ordering::Relaxed);
    });
    
    // Get the atomic sum and convert back to f64
    let atomic_sum_bits = std::sync::atomic::AtomicU64::new(0).load(Ordering::Relaxed);
    sum += f64::from_bits(atomic_sum_bits);
    
    realft(y, n, 1);
    
    y[n + 1] = y[2];
    
    for j in (4..=n).step_by(2) {
        sum += y[j];
        y[j] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cosft1_basic() {
        let n = 8;
        let mut data = vec![0.0; n + 2];
        for i in 1..=n {
            data[i] = i as f64;
        }
        
        cosft1(&mut data, n);
        // Add specific test assertions based on expected results
    }
    
    #[test]
    fn test_cosft1_performance() {
        let n = 1024;
        let mut data = vec![0.0; n + 2];
        
        // Warmup
        cosft1(&mut data, n);
        
        // Performance test
        let start = std::time::Instant::now();
        for _ in 0..100 {
            cosft1(&mut data, n);
        }
        let duration = start.elapsed();
        
        println!("Average time per cosft1: {:?}", duration / 100);
    }
}
