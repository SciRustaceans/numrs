use std::f64::consts::PI;
use rayon::prelude::*;
use ndarray::{Array1, ArrayView1, ArrayViewMut1, s};
use thiserror::Error;

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

    // For small arrays, use direct computation (more efficient)
    if n <= 32 {
        return Ok(correl_direct(data1, data2));
    }

    // Compute FFTs in parallel
    let (fft1, fft2) = compute_parallel_ffts(data1, data2, n);

    // Process frequency domain with optimization
    let mut ans_fft = process_correlation_frequency_domain(&fft1, &fft2, n);

    // Inverse FFT to get correlation in time domain
    realft(&mut ans_fft, n, -1);

    // Extract the real part (correlation result)
    Ok(extract_correlation_result(&ans_fft, n))
}

/// Direct correlation computation for small arrays (more efficient than FFT)
fn correl_direct(data1: &[f64], data2: &[f64]) -> Array1<f64> {
    let n = data1.len();
    let mut result = Array1::zeros(n);
    
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += data1[i + lag] * data2[i];
        }
        result[lag] = sum;
    }
    
    result
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
    let scale = 1.0 / no2 as f64;
    let mut ans_fft = Array1::zeros(2 * n);

    // Process complex pairs in parallel with optimized patterns
    ans_fft
        .par_chunks_mut(2)
        .zip(fft1.par_chunks(2))
        .zip(fft2.par_chunks(2))
        .enumerate()
        .for_each(|(i, ((ans_chunk, fft1_chunk), fft2_chunk))| {
            if i >= 1 && i < no2 + 1 {
                let fft1_re = fft1_chunk[0];
                let fft1_im = fft1_chunk[1];
                let fft2_re = fft2_chunk[0];
                let fft2_im = fft2_chunk[1];

                // Cross-correlation in frequency domain: FFT1 * conjugate(FFT2)
                let corr_re = fft1_re * fft2_re + fft1_im * fft2_im; // Real part
                let corr_im = fft1_im * fft2_re - fft1_re * fft2_im; // Imaginary part

                // Scale and store
                ans_chunk[0] = corr_re * scale;
                ans_chunk[1] = corr_im * scale;
            }
        });

    // Handle DC and Nyquist components
    ans_fft[0] = (fft1[0] * fft2[0]) * scale; // DC component
    ans_fft[1] = 0.0; // Nyquist component (real signal)
    
    // Handle the Nyquist frequency component for even n
    if n % 2 == 0 {
        ans_fft[n] = (fft1[1] * fft2[1]) * scale;
    }

    ans_fft
}

/// Manual SIMD-like optimization using f64x2 pattern
fn process_correlation_frequency_domain_optimized(
    fft1: &Array1<f64>,
    fft2: &Array1<f64>,
    n: usize,
) -> Array1<f64> {
    let no2 = n >> 1;
    let scale = 1.0 / no2 as f64;
    let mut ans_fft = Array1::zeros(2 * n);

    // Process pairs of complex numbers together for better cache locality
    for i in 1..no2 {
        let idx1 = 2 * i;
        let idx2 = 2 * (i + 1);
        
        // Process two complex numbers at once
        let (fft1_re1, fft1_im1) = (fft1[idx1], fft1[idx1 + 1]);
        let (fft2_re1, fft2_im1) = (fft2[idx1], fft2[idx1 + 1]);
        
        let (fft1_re2, fft1_im2) = (fft1[idx2], fft1[idx2 + 1]);
        let (fft2_re2, fft2_im2) = (fft2[idx2], fft2[idx2 + 1]);
        
        // Compute complex multiplication: fft1 * conjugate(fft2)
        let corr_re1 = fft1_re1 * fft2_re1 + fft1_im1 * fft2_im1;
        let corr_im1 = fft1_im1 * fft2_re1 - fft1_re1 * fft2_im1;
        
        let corr_re2 = fft1_re2 * fft2_re2 + fft1_im2 * fft2_im2;
        let corr_im2 = fft1_im2 * fft2_re2 - fft1_re2 * fft2_im2;
        
        // Scale and store
        ans_fft[idx1] = corr_re1 * scale;
        ans_fft[idx1 + 1] = corr_im1 * scale;
        ans_fft[idx2] = corr_re2 * scale;
        ans_fft[idx2 + 1] = corr_im2 * scale;
    }

    // Handle remaining elements
    if no2 % 2 == 1 {
        let i = no2;
        let idx = 2 * i;
        let (fft1_re, fft1_im) = (fft1[idx], fft1[idx + 1]);
        let (fft2_re, fft2_im) = (fft2[idx], fft2[idx + 1]);
        
        let corr_re = fft1_re * fft2_re + fft1_im * fft2_im;
        let corr_im = fft1_im * fft2_re - fft1_re * fft2_im;
        
        ans_fft[idx] = corr_re * scale;
        ans_fft[idx + 1] = corr_im * scale;
    }

    // Handle DC and Nyquist components
    ans_fft[0] = (fft1[0] * fft2[0]) * scale;
    ans_fft[1] = 0.0;
    
    if n % 2 == 0 {
        ans_fft[n] = (fft1[1] * fft2[1]) * scale;
    }

    ans_fft
}

/// Extract the real correlation result from the complex array
fn extract_correlation_result(ans_fft: &Array1<f64>, n: usize) -> Array1<f64> {
    let mut result = Array1::zeros(n);
    
    // The correlation result is in the real part of the complex array
    // For real signals, the FFT result is stored in a packed format
    result[0] = ans_fft[0]; // DC component
    
    for i in 1..n {
        result[i] = ans_fft[2 * i];
    }
    
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

    // Compute means and standard deviations in parallel
    let (mean1, mean2, std1, std2) = rayon::join(
        || data1.iter().sum::<f64>() / n as f64,
        || data2.iter().sum::<f64>() / n as f64,
        || {
            let mean1 = data1.iter().sum::<f64>() / n as f64;
            (data1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / n as f64).sqrt()
        },
        || {
            let mean2 = data2.iter().sum::<f64>() / n as f64;
            (data2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / n as f64).sqrt()
        },
    );

    // Check for zero standard deviation
    if std1 == 0.0 || std2 == 0.0 {
        return Err(CorrelError::ZeroStdDev);
    }

    // Normalize inputs
    let data1_norm: Vec<f64> = data1.iter().map(|&x| (x - mean1) / std1).collect();
    let data2_norm: Vec<f64> = data2.iter().map(|&x| (x - mean2) / std2).collect();

    // Compute correlation of normalized signals
    correl(&data1_norm, &data2_norm)
}

/// Fast normalized correlation using precomputed statistics
pub fn correl_normalized_fast(data1: &[f64], data2: &[f64]) -> Result<Array1<f64>, CorrelError> {
    let n = data1.len();
    if n == 0 {
        return Err(CorrelError::EmptyInput);
    }
    if data2.len() != n {
        return Err(CorrelError::LengthMismatch);
    }

    // Precompute statistics with single pass
    let (sum1, sum2, sum_sq1, sum_sq2, sum_prod) = data1.iter()
        .zip(data2.iter())
        .fold((0.0, 0.0, 0.0, 0.0, 0.0), |(s1, s2, sq1, sq2, sp), (&x, &y)| {
            (s1 + x, s2 + y, sq1 + x * x, sq2 + y * y, sp + x * y)
        });

    let mean1 = sum1 / n as f64;
    let mean2 = sum2 / n as f64;
    let std1 = ((sum_sq1 / n as f64) - mean1 * mean1).sqrt();
    let std2 = ((sum_sq2 / n as f64) - mean2 * mean2).sqrt();

    if std1 == 0.0 || std2 == 0.0 {
        return Err(CorrelError::ZeroStdDev);
    }

    // For small n, use direct computation
    if n <= 32 {
        let mut result = Array1::zeros(n);
        let norm_factor = 1.0 / (std1 * std2 * n as f64);
        
        for lag in 0..n {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += (data1[i + lag] - mean1) * (data2[i] - mean2);
            }
            result[lag] = sum * norm_factor;
        }
        return Ok(result);
    }

    // For larger n, use FFT-based method
    let data1_norm: Vec<f64> = data1.iter().map(|&x| (x - mean1) / std1).collect();
    let data2_norm: Vec<f64> = data2.iter().map(|&x| (x - mean2) / std2).collect();
    
    correl(&data1_norm, &data2_norm)
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

/// Efficient auto-correlation with symmetry optimization
pub fn autocorrel_fast(data: &[f64]) -> Result<Array1<f64>, CorrelError> {
    let n = data.len();
    if n == 0 {
        return Err(CorrelError::EmptyInput);
    }

    // For small arrays, use direct computation
    if n <= 32 {
        let mut result = Array1::zeros(n);
        
        for lag in 0..n {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += data[i + lag] * data[i];
            }
            result[lag] = sum;
        }
        return Ok(result);
    }

    // For larger arrays, use FFT with symmetry optimization
    let fft_data = compute_real_fft(data, n);
    let mut auto_fft = Array1::zeros(2 * n);
    
    // Auto-correlation: FFT * conjugate(FFT) = |FFT|²
    auto_fft[0] = fft_data[0] * fft_data[0]; // DC component
    
    for i in 1..(n/2 + 1) {
        let idx = 2 * i;
        let re = fft_data[idx];
        let im = fft_data[idx + 1];
        auto_fft[idx] = re * re + im * im; // Power spectrum
        auto_fft[idx + 1] = 0.0;
    }
    
    realft(&mut auto_fft, n, -1);
    Ok(extract_correlation_result(&auto_fft, n))
}

/// Wrapper for realft function (placeholder implementation)
fn realft(data: &mut Array1<f64>, n: usize, isign: i32) {
    // Simple DFT implementation for demonstration
    // Replace with your optimized realft implementation
    if isign == 1 {
        // Forward FFT
        dft_forward(data, n);
    } else {
        // Inverse FFT
        dft_inverse(data, n);
    }
}

/// Simple DFT implementation for demonstration
fn dft_forward(data: &mut Array1<f64>, n: usize) {
    let mut temp = data.to_owned();
    
    for k in 0..n {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        
        for j in 0..n {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let (sin_angle, cos_angle) = angle.sin_cos();
            sum_re += data[j] * cos_angle;
            sum_im += data[j] * sin_angle;
        }
        
        temp[2 * k] = sum_re;
        if k > 0 && k < n / 2 {
            temp[2 * k + 1] = sum_im;
        }
    }
    
    data.assign(&temp);
}

/// Simple inverse DFT implementation for demonstration
fn dft_inverse(data: &mut Array1<f64>, n: usize) {
    let mut temp = Array1::zeros(n);
    let scale = 1.0 / n as f64;
    
    for j in 0..n {
        let mut sum = 0.0;
        
        for k in 0..n {
            let angle = 2.0 * PI * k as f64 * j as f64 / n as f64;
            let (sin_angle, cos_angle) = angle.sin_cos();
            if k == 0 {
                sum += data[0] * cos_angle;
            } else if k <= n / 2 {
                sum += data[2 * k] * cos_angle - data[2 * k + 1] * sin_angle;
            }
        }
        
        temp[j] = sum * scale;
    }
    
    for j in 0..n {
        data[j] = temp[j];
    }
}

/// Error types for correlation computation
#[derive(Debug, Error)]
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
            (vec![1.0, 2.0].as_slice(), vec![1.0, 2.0].as_slice()),
            (vec![3.0, 4.0].as_slice(), vec![3.0, 4.0].as_slice()),
        ];
        
        let results = correl_batch(&pairs)?;
        
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

    #[test]
    fn test_direct_correlation_small() {
        let data1 = vec![1.0, 2.0];
        let data2 = vec![1.0, 2.0];
        
        let result = correl_direct(&data1, &data2);
        
        assert_relative_eq!(result[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fast_normalized_correlation() -> Result<(), CorrelError> {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = correl_normalized_fast(&data1, &data2)?;
        
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        Ok(())
    }
}
