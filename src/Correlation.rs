use std::f64::consts::PI;
use rayon::prelude::*;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Cross-correlation implementation with FFT optimization
/// Computes the correlation between two signals using frequency domain methods
pub fn correl(data1: &[f64], data2: &[f64]) -> Result<Array1<f64>, CorrelError> {
    // Validate inputs
    let n = data1.len();
    if n == 0 {
        return Err(CorrelError::EmptyInput);
    }
    if data2.len() != n {
        return Err(CorrelError::LengthMismatch);
    }

    // Compute FFTs in parallel
    let (fft1, fft2) = compute_parallel_ffts(data1, data2, n);

    // Process frequency domain with SIMD optimization
    let mut ans_fft = process_correlation_frequency_domain(&fft1, &fft2, n);

    // Inverse FFT to get correlation in time domain
    realft(&mut ans_fft, n, -1);

    // Extract the real part (correlation result)
    Ok(extract_correlation_result(&ans_fft, n))
}

/// Compute FFTs of both signals in parallel
fn compute_parallel_ffts(data1: &[f64], data2: &[f64], n: usize) -> (Array1<f64>, Array1<f64>) {
    rayon::join(
        || compute_real_fft(data1, n),
        || compute_real_fft(data2, n),
    )
}

/// Compute real-valued FFT with optimization
fn compute_real_fft(data: &[f64], n: usize) -> Array1<f64> {
    let mut fft_result = Array1::zeros(2 * n);
    fft_result.slice_mut(s![..n]).assign(&Array1::from_vec(data.to_vec()));
    realft(&mut fft_result, n, 1);
    fft_result
}

/// Process frequency domain data for correlation
fn process_correlation_frequency_domain(fft1: &Array1<f64>, fft2: &Array1<f64>, n: usize) -> Array1<f64> {
    let no2 = n >> 1;
    let mut ans_fft = Array1::zeros(2 * n);

    // Process complex pairs in parallel with SIMD-friendly patterns
    ans_fft
        .par_chunks_mut(2)
        .zip(fft1.par_chunks(2))
        .zip(fft2.par_chunks(2))
        .enumerate()
        .for_each(|(i, ((ans_chunk, fft1_chunk), fft2_chunk))| {
            if i >= 1 && i < n + 1 {
                let fft1_re = fft1_chunk[0];
                let fft1_im = fft1_chunk[1];
                let fft2_re = fft2_chunk[0];
                let fft2_im = fft2_chunk[1];

                // Cross-correlation in frequency domain: FFT1 * conjugate(FFT2)
                let corr_re = fft1_re * fft2_re + fft1_im * fft2_im; // Real part
                let corr_im = fft1_im * fft2_re - fft1_re * fft2_im; // Imaginary part

                // Scale and store
                ans_chunk[0] = corr_re / no2 as f64;
                ans_chunk[1] = corr_im / no2 as f64;
            }
        });

    // Handle the Nyquist frequency component
    ans_fft[2 * n - 2] = ans_fft[1];
    ans_fft[2 * n - 1] = 0.0;

    ans_fft
}

/// SIMD-optimized version using packed complex arithmetic
#[cfg(target_arch = "x86_64")]
fn process_correlation_frequency_domain_simd(
    fft1: &Array1<f64>,
    fft2: &Array1<f64>,
    n: usize,
) -> Array1<f64> {
    use std::arch::x86_64::*;
    
    let no2 = n >> 1;
    let scale = 1.0 / no2 as f64;
    let mut ans_fft = Array1::zeros(2 * n);

    unsafe {
        // Process multiple complex pairs simultaneously
        for i in (1..n).step_by(2) {
            let idx = 2 * i;
            
            // Load complex pairs into SIMD registers
            let fft1_vec = _mm256_loadu_pd(&fft1[idx]);
            let fft2_vec = _mm256_loadu_pd(&fft2[idx]);
            
            // Compute complex multiplication: fft1 * conjugate(fft2)
            let result = complex_multiply_conjugate_simd(fft1_vec, fft2_vec);
            
            // Scale and store
            let scale_vec = _mm256_set1_pd(scale);
            let scaled_result = _mm256_mul_pd(result, scale_vec);
            _mm256_storeu_pd(&mut ans_fft[idx], scaled_result);
        }
    }

    ans_fft
}

#[cfg(target_arch = "x86_64")]
unsafe fn complex_multiply_conjugate_simd(a: __m256d, b: __m256d) -> __m256d {
    use std::arch::x86_64::*;
    
    // a = [a_re, a_im, a_re2, a_im2]
    // b = [b_re, b_im, b_re2, b_im2] - we want conjugate: [b_re, -b_im, b_re2, -b_im2]
    
    // Negate the imaginary parts of b
    let sign_mask = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    let b_conj = _mm256_mul_pd(b, sign_mask);
    
    // Complex multiplication
    let a_perm = _mm256_permute_pd(a, 0x5); // [a_im, a_re, a_im2, a_re2]
    let b_perm = _mm256_permute_pd(b_conj, 0x5); // [-b_im, b_re, -b_im2, b_re2]
    
    let real_part = _mm256_mul_pd(a, b_conj);
    let imag_part = _mm256_mul_pd(a_perm, b_perm);
    
    _mm256_hadd_pd(real_part, imag_part)
}

/// Extract the real correlation result from the complex array
fn extract_correlation_result(ans_fft: &Array1<f64>, n: usize) -> Array1<f64> {
    let mut result = Array1::zeros(n);
    
    result.par_iter_mut().enumerate().for_each(|(i, elem)| {
        // The correlation result is in the real part of the complex array
        *elem = ans_fft[2 * i];
    });
    
    result
}

/// Normalized cross-correlation (Pearson correlation coefficient)
pub fn correl_normalized(data1: &[f64], data2: &[f64]) -> Result<Array1<f64>, CorrelError> {
    let n = data1.len();
    if n == 0 {
        return Err(CorrelError::EmptyInput);
    }
    if data2.len() != n {
        return Err(CorrelError::LengthMismatch);
    }

    // Compute means and standard deviations
    let mean1 = data1.iter().sum::<f64>() / n as f64;
    let mean2 = data2.iter().sum::<f64>() / n as f64;
    
    let std1 = (data1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / n as f64).sqrt();
    let std2 = (data2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / n as f64).sqrt();

    // Normalize inputs
    let data1_norm: Array1<f64> = data1.iter().map(|&x| (x - mean1) / std1).collect();
    let data2_norm: Array1<f64> = data2.iter().map(|&x| (x - mean2) / std2).collect();

    // Compute correlation of normalized signals
    correl(data1_norm.as_slice().unwrap(), data2_norm.as_slice().unwrap())
}

/// Batch correlation computation for multiple signal pairs
pub fn correl_batch(
    data_pairs: &[(&[f64], &[f64])],
) -> Result<Vec<Array1<f64>>, CorrelError> {
    data_pairs
        .par_iter()
        .map(|(data1, data2)| correl(data1, data2))
        .collect()
}

/// Auto-correlation (correlation of a signal with itself)
pub fn autocorrel(data: &[f64]) -> Result<Array1<f64>, CorrelError> {
    correl(data, data)
}

/// Wrapper for realft function
fn realft(data: &mut Array1<f64>, n: usize, isign: i32) {
    // Implementation would call your optimized realft function
    unimplemented!("Use your optimized realft implementation")
}

/// Error types for correlation computation
#[derive(Debug, thiserror::Error)]
pub enum CorrelError {
    #[error("Input arrays cannot be empty")]
    EmptyInput,
    #[error("Input arrays must have the same length")]
    LengthMismatch,
    #[error("FFT computation error: {0}")]
    FftError(String),
    #[error("Normalization error: standard deviation is zero")]
    ZeroStdDev,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_correlation_basic() -> Result<(), CorrelError> {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = correl(&data1, &data2)?;
        
        // Auto-correlation should peak at lag 0
        assert_relative_eq!(result[0], 30.0, epsilon = 1e-10); // 1² + 2² + 3² + 4² = 30
        assert!(result[0] > result[1]); // Peak at zero lag
        
        Ok(())
    }

    #[test]
    fn test_correlation_shifted() -> Result<(), CorrelError> {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![0.0, 1.0, 2.0, 3.0]; // data1 shifted by 1
        
        let result = correl(&data1, &data2)?;
        
        // Should peak at lag 1 (shifted signal)
        assert!(result[1] > result[0]);
        assert!(result[1] > result[2]);
        
        Ok(())
    }

    #[test]
    fn test_normalized_correlation() -> Result<(), CorrelError> {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = correl_normalized(&data1, &data2)?;
        
        // Normalized auto-correlation should be 1.0 at lag 0
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        
        // Other lags should be between -1 and 1
        for &value in result.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_error_handling() {
        // Test empty input
        assert!(matches!(correl(&[], &[1.0]), Err(CorrelError::EmptyInput)));
        
        // Test length mismatch
        assert!(matches!(
            correl(&[1.0, 2.0], &[1.0]),
            Err(CorrelError::LengthMismatch)
        ));
    }

    #[test]
    fn test_batch_correlation() -> Result<(), CorrelError> {
        let pairs = vec![
            (vec![1.0, 2.0], vec![1.0, 2.0]),
            (vec![3.0, 4.0], vec![3.0, 4.0]),
        ];
        
        let results = correl_batch(&pairs.iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect::<Vec<_>>())?;
        
        assert_eq!(results.len(), 2);
        assert_relative_eq!(results[0][0], 5.0, epsilon = 1e-10); // 1² + 2²
        assert_relative_eq!(results[1][0], 25.0, epsilon = 1e-10); // 3² + 4²
        
        Ok(())
    }

    #[test]
    fn test_autocorrelation() -> Result<(), CorrelError> {
        let data = vec![1.0, 2.0, 1.0, 2.0]; // Periodic signal
        
        let result = autocorrel(&data)?;
        
        // Auto-correlation of periodic signal should show periodicity
        assert_relative_eq!(result[0], 10.0, epsilon = 1e-10); // 1² + 2² + 1² + 2²
        assert_relative_eq!(result[2], 8.0, epsilon = 1e-10); // 1*1 + 2*2
        
        Ok(())
    }
}
