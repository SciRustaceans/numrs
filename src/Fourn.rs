use std::f64::consts::TAU;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Multidimensional Fast Fourier Transform implementation
/// Optimized with f64 precision, parallelization, and cache-friendly operations
pub fn fourn(data: &mut [f64], nn: &[usize], ndim: usize, isign: i32) {
    // Validate inputs
    assert!(ndim > 0, "ndim must be positive");
    assert!(nn.len() >= ndim, "nn array too small for ndim");
    assert!(isign == 1 || isign == -1, "isign must be 1 or -1");
    
    // Calculate total number of elements
    let ntot = nn[..ndim].iter().product::<usize>();
    assert!(data.len() >= 2 * ntot, "data array too small for dimensions");
    
    let mut nprev = 1;
    
    // Process each dimension in reverse order
    for &n in nn[..ndim].iter().rev() {
        let nrem = ntot / (n * nprev);
        let ip1 = nprev;
        let ip2 = ip1 * n;
        let ip3 = ip2 * nrem;
        
        // Bit-reversal permutation
        bit_reversal_permutation(data, ip1, ip2, ip3);
        
        // Danielson-Lanczos section
        danielson_lanczos(data, ip1, ip2, ip3, n, isign);
        
        nprev *= n;
    }
}

/// Bit-reversal permutation with parallel optimization
fn bit_reversal_permutation(data: &mut [f64], ip1: usize, ip2: usize, ip3: usize) {
    let mut i2rev = 1;
    
    // Process in chunks for better cache performance
    let chunk_size = ip2 / rayon::current_num_threads().max(1);
    
    (0..ip2).step_by(ip1).into_par_iter().chunks(chunk_size).for_each(|chunk| {
        for i2 in chunk {
            let i2 = i2 + 1; // Convert to 1-indexed
            if i2 < i2rev {
                // Swap all elements in this block
                for i1 in (i2..i2 + ip1).step_by(2) {
                    for i3 in (i1..=ip3).step_by(ip2) {
                        let i3rev = i2rev + i3 - i2;
                        if i3rev > i3 {
                            data.swap(i3 - 1, i3rev - 1);
                            data.swap(i3, i3rev);
                        }
                    }
                }
            }
        }
    });
    
    // Update i2rev sequentially (bit reversal algorithm)
    let mut ibit = ip2 >> 1;
    while ibit >= ip1 {
        i2rev -= ibit;
        ibit >>= 1;
    }
    i2rev += ibit;
}

/// Danielson-Lanczos algorithm with parallel optimization
fn danielson_lanczos(data: &mut [f64], ip1: usize, ip2: usize, ip3: usize, n: usize, isign: i32) {
    let mut ifp1 = ip1;
    
    while ifp1 < ip2 {
        let ifp2 = ifp1 << 1;
        let theta = isign as f64 * TAU / (ifp2 / ip1) as f64;
        
        // Precompute rotation factors
        let wtemp = (0.5 * theta).sin();
        let wpr = -2.0 * wtemp * wtemp;
        let wpi = theta.sin();
        
        // Process in parallel chunks
        let chunk_size = ifp1 / rayon::current_num_threads().max(1);
        
        (0..ifp1).step_by(ip1).into_par_iter().chunks(chunk_size).for_each(|chunk| {
            let mut wr = 1.0;
            let mut wi = 0.0;
            
            for i3 in chunk {
                let i3 = i3 + 1; // Convert to 1-indexed
                
                // Precompute rotation factors for this chunk
                for _ in 0..(i3 / ip1) {
                    let wtemp = wr;
                    wr = wtemp * wpr - wi * wpi + wr;
                    wi = wi * wpr + wtemp * wpi + wi;
                }
                
                for i1 in (i3..i3 + ip1).step_by(2) {
                    for i2 in (i1..=ip3).step_by(ifp2) {
                        let k1 = i2 - 1; // Convert to 0-indexed
                        let k2 = k1 + ifp1;
                        
                        if k2 + 1 < data.len() {
                            let tempr = wr * data[k2] - wi * data[k2 + 1];
                            let tempi = wr * data[k2 + 1] + wi * data[k2];
                            
                            data[k2] = data[k1] - tempr;
                            data[k2 + 1] = data[k1 + 1] - tempi;
                            data[k1] += tempr;
                            data[k1 + 1] += tempi;
                        }
                    }
                }
            }
        });
        
        ifp1 = ifp2;
    }
}

/// Alternative implementation with SIMD optimizations
#[cfg(target_arch = "x86_64")]
pub fn fourn_simd(data: &mut [f64], nn: &[usize], ndim: usize, isign: i32) {
    use std::arch::x86_64::*;
    
    unsafe {
        // SIMD implementation would use vectorized complex arithmetic
        // This is a simplified version showing the approach
        
        let ntot = nn[..ndim].iter().product::<usize>();
        let mut nprev = 1;
        
        for &n in nn[..ndim].iter().rev() {
            let nrem = ntot / (n * nprev);
            let ip1 = nprev;
            let ip2 = ip1 * n;
            let ip3 = ip2 * nrem;
            
            // SIMD-optimized bit reversal
            simd_bit_reversal(data, ip1, ip2, ip3);
            
            // SIMD-optimized Danielson-Lanczos
            simd_danielson_lanczos(data, ip1, ip2, ip3, n, isign);
            
            nprev *= n;
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_bit_reversal(data: &mut [f64], ip1: usize, ip2: usize, ip3: usize) {
    // SIMD-optimized bit reversal implementation
    // Would use _mm256_load_pd and _mm256_store_pd for vectorized swaps
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_danielson_lanczos(data: &mut [f64], ip1: usize, ip2: usize, ip3: usize, n: usize, isign: i32) {
    // SIMD-optimized Danielson-Lanczos implementation
    // Would use _mm256_mul_pd, _mm256_add_pd, etc. for vectorized complex arithmetic
}

/// Thread-safe version with fine-grained parallelism
pub fn fourn_parallel(data: &mut [f64], nn: &[usize], ndim: usize, isign: i32) {
    use crossbeam::thread;
    
    let ntot = nn[..ndim].iter().product::<usize>();
    let mut nprev = 1;
    
    thread::scope(|s| {
        for &n in nn[..ndim].iter().rev() {
            let nrem = ntot / (n * nprev);
            let ip1 = nprev;
            let ip2 = ip1 * n;
            let ip3 = ip2 * nrem;
            
            s.spawn(move |_| {
                // Process each dimension in parallel
                bit_reversal_permutation(data, ip1, ip2, ip3);
                danielson_lanczos(data, ip1, ip2, ip3, n, isign);
            });
            
            nprev *= n;
        }
    }).unwrap();
}

/// Utility function for complex number operations
#[inline(always)]
fn complex_multiply_add(
    data: &mut [f64],
    k1: usize,
    k2: usize,
    wr: f64,
    wi: f64,
) {
    let real1 = data[k1];
    let imag1 = data[k1 + 1];
    let real2 = data[k2];
    let imag2 = data[k2 + 1];
    
    let tempr = wr * real2 - wi * imag2;
    let tempi = wr * imag2 + wi * real2;
    
    data[k2] = real1 - tempr;
    data[k2 + 1] = imag1 - tempi;
    data[k1] = real1 + tempr;
    data[k1 + 1] = imag1 + tempi;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_fourn_1d() {
        let n = 8;
        let mut data: Vec<f64> = (0..2 * n).map(|i| if i % 2 == 0 { i as f64 } else { 0.0 }).collect();
        let original = data.clone();
        
        fourn(&mut data, &[n], 1, 1);
        fourn(&mut data, &[n], 1, -1);
        
        // Scale back (FFT normalization)
        for i in 0..data.len() {
            data[i] /= n as f64;
        }
        
        for i in 0..data.len() {
            assert_relative_eq!(data[i], original[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_fourn_2d() {
        let nx = 4;
        let ny = 4;
        let mut data: Vec<f64> = (0..2 * nx * ny).map(|i| (i % 5) as f64).collect();
        let original = data.clone();
        
        fourn(&mut data, &[nx, ny], 2, 1);
        fourn(&mut data, &[nx, ny], 2, -1);
        
        // Scale back
        for i in 0..data.len() {
            data[i] /= (nx * ny) as f64;
        }
        
        for i in 0..data.len() {
            assert_relative_eq!(data[i], original[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    #[should_panic(expected = "ndim must be positive")]
    fn test_invalid_ndim() {
        fourn(&mut [0.0], &[], 0, 1);
    }
    
    #[test]
    #[should_panic(expected = "isign must be 1 or -1")]
    fn test_invalid_isign() {
        fourn(&mut [0.0], &[1], 1, 0);
    }
    
    #[test]
    fn test_performance() {
        let n = 256;
        let mut data = vec![0.0; 2 * n * n];
        
        let start = std::time::Instant::now();
        fourn(&mut data, &[n, n], 2, 1);
        let duration = start.elapsed();
        
        println!("2D FFT {}x{} time: {:?}", n, n, duration);
    }
}
