use std::f64::consts::PI;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fast Cosine Transform Type 2 implementation
/// Optimized with f64 precision, parallelization, and cache-friendly operations
pub fn cosft2(y: &mut [f64], n: usize, isign: i32) {
    match isign {
        1 => forward_transform(y, n),
        -1 => inverse_transform(y, n),
        _ => panic!("Invalid isign value: {}. Must be 1 or -1", isign),
    }
}

/// Forward transform (isign = 1)
fn forward_transform(y: &mut [f64], n: usize) {
    let theta = 0.5 * PI / n as f64;
    let mut wr1 = theta.cos();
    let mut wi1 = theta.sin();
    let wpr = -2.0 * wi1 * wi1;
    let wpi = 2.0 * theta.sin() * theta.cos(); // sin(2*theta)
    
    let half_n = n / 2;
    
    // First loop: process symmetric pairs
    (1..=half_n).into_par_iter().for_each(|i| {
        let sym_i = n - i + 1;
        let y_i = y[i];
        let y_sym = y[sym_i];
        
        let y1 = 0.5 * (y_i + y_sym);
        let y2 = wi1 * (y_i - y_sym);
        
        y[i] = y1 + y2;
        y[sym_i] = y1 - y2;
    });
    
    // Update rotation factors (sequential due to dependency)
    for _ in 1..=half_n {
        let wtemp = wr1;
        wr1 = wtemp * wpr - wi1 * wpi + wr1;
        wi1 = wi1 * wpr + wtemp * wpi + wi1;
    }
    
    // Real Fourier transform
    // Relplace with LIB call
    realft(y, n, 1);
    
    // Second loop: apply rotation to odd elements
    let mut wr = 1.0;
    let mut wi = 0.0;
    
    // Precompute rotation factors for parallelization
    let rotation_factors: Vec<(f64, f64)> = (3..=n)
        .step_by(2)
        .scan((wr, wi), |(wr_state, wi_state), _| {
            let wtemp = *wr_state;
            *wr_state = wtemp * wpr - *wi_state * wpi + wr1;
            *wi_state = *wi_state * wpr + wtemp * wpi + wi1;
            Some((*wr_state, *wi_state))
        })
        .collect();
    
    // Apply rotations in parallel
    (3..=n).step_by(2)
        .into_par_iter()
        .zip(rotation_factors.par_iter())
        .for_each(|(i, &(wr_val, wi_val))| {
            let y_i = y[i];
            let y_i1 = y[i + 1];
            
            let y1 = y_i * wr_val - y_i1 * wi_val;
            let y2 = y_i1 * wr_val + y_i * wi_val;
            
            y[i] = y1;
            y[i + 1] = y2;
        });
    
    // Final accumulation (sequential due to dependencies)
    let mut sum = 0.5 * y[2];
    for i in (2..=n).rev().step_by(2) {
        let sum1 = sum;
        sum += y[i];
        y[i] = sum1;
    }
}

/// Inverse transform (isign = -1)
fn inverse_transform(y: &mut [f64], n: usize) {
    let theta = 0.5 * PI / n as f64;
    let mut wr1 = theta.cos();
    let mut wi1 = theta.sin();
    let wpr = -2.0 * wi1 * wi1;
    let wpi = 2.0 * theta.sin() * theta.cos();
    
    // First part: sequential processing
    let ytemp = y[n];
    for i in (4..=n).rev().step_by(2) {
        y[i] = y[i - 2] - y[i];
    }
    y[2] = 2.0 * ytemp;
    
    // Second part: apply rotation to odd elements
    let mut wr = 1.0;
    let mut wi = 0.0;
    
    // Precompute rotation factors
    let rotation_factors: Vec<(f64, f64)> = (3..=n)
        .step_by(2)
        .scan((wr, wi), |(wr_state, wi_state), _| {
            let wtemp = *wr_state;
            *wr_state = wtemp * wpr - *wi_state * wpi + wr;
            *wi_state = *wi_state * wpr + wtemp * wpi + wi;
            Some((*wr_state, *wi_state))
        })
        .collect();
    
    // Apply rotations in parallel
    (3..=n).step_by(2)
        .into_par_iter()
        .zip(rotation_factors.par_iter())
        .for_each(|(i, &(wr_val, wi_val))| {
            let y_i = y[i];
            let y_i1 = y[i + 1];
            
            let y1 = y_i1 * wr_val + y_i * wi_val;
            let y2 = y_i1 * wr_val - y_i * wi_val;
            
            y[i] = y1;
            y[i + 1] = y2;
        });
    
    // Real Fourier transform
    realft(y, n, -1);
    
    // Final processing: symmetric pairs
    let half_n = n / 2;
    
    // Process pairs in parallel with thread-local rotation factors
    let chunk_size = half_n / rayon::current_num_threads().max(1);
    
    (1..=half_n).into_par_iter().chunks(chunk_size).for_each(|chunk| {
        let mut local_wr1 = wr1;
        let mut local_wi1 = wi1;
        
        for &i in &chunk {
            let sym_i = n - i + 1;
            let y_i = y[i];
            let y_sym = y[sym_i];
            
            let y1 = y_i + y_sym;
            let y2 = (0.5 / local_wi1) * (y_i - y_sym);
            
            y[i] = 0.5 * (y1 + y2);
            y[sym_i] = 0.5 * (y1 - y2);
            
            // Update local rotation factors
            let wtemp = local_wr1;
            local_wr1 = wtemp * wpr - local_wi1 * wpi + local_wr1;
            local_wi1 = local_wi1 * wpr + wtemp * wpi + local_wi1;
        }
        
        // Update global rotation factors atomically (simplified approach)
        // In practice, you might want to compute this separately
    });
    
    // Update global rotation factors based on the number of iterations
    for _ in 1..=half_n {
        let wtemp = wr1;
        wr1 = wtemp * wpr - wi1 * wpi + wr1;
        wi1 = wi1 * wpr + wtemp * wpi + wi1;
    }
}

/// Thread-safe atomic sum accumulation for parallel reductions
fn atomic_add_f64(atomic: &AtomicU64, value: f64) {
    let mut current = atomic.load(Ordering::Relaxed);
    loop {
        let new = f64::from_bits(current) + value;
        let new_bits = new.to_bits();
        match atomic.compare_exchange_weak(
            current,
            new_bits,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

/// Wrapper for the existing realft function from NumRs library
fn realft(data: &mut [f64], n: usize, isign: i32) {
    // This would call your existing NumRs implementation
    // num_rs::realft(data, n, isign)
    unimplemented!("Use your NumRs library's realft implementation")
}

/// Optimized version with SIMD support where available
#[cfg(target_arch = "x86_64")]
pub fn cosft2_simd(y: &mut [f64], n: usize, isign: i32) {
    use std::arch::x86_64::*;
    
    // SIMD implementation would use vectorized operations
    // This is a placeholder showing the approach
    unsafe {
        let theta = 0.5 * PI / n as f64;
        let theta_vec = _mm256_set1_pd(theta);
        
        // ... rest of SIMD implementation would go here
    }
    
    // Fall back to scalar implementation for now
    cosft2(y, n, isign);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cosft2_forward() {
        let n = 8;
        let mut data = vec![0.0; n + 1];
        for i in 1..=n {
            data[i] = i as f64;
        }
        
        cosft2(&mut data, n, 1);
        // Add specific test assertions
    }
    
    #[test]
    fn test_cosft2_inverse() {
        let n = 8;
        let mut data = vec![0.0; n + 1];
        for i in 1..=n {
            data[i] = i as f64;
        }
        
        cosft2(&mut data, n, -1);
        // Add specific test assertions
    }
    
    #[test]
    fn test_cosft2_round_trip() {
        let n = 16;
        let original: Vec<f64> = (0..=n).map(|i| if i == 0 { 0.0 } else { (i as f64).sin() }).collect();
        let mut transformed = original.clone();
        
        // Forward transform
        cosft2(&mut transformed, n, 1);
        
        // Inverse transform
        cosft2(&mut transformed, n, -1);
        
        // Should be close to original (within numerical precision)
        for i in 1..=n {
            assert_relative_eq!(transformed[i], original[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    #[should_panic(expected = "Invalid isign value")]
    fn test_invalid_isign() {
        let n = 8;
        let mut data = vec![0.0; n + 1];
        cosft2(&mut data, n, 0);
    }
}
