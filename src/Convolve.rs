use std::f64::consts::PI;
use rayon::prelude::*;
use ndarray::{Array1, ArrayView, ArrayViewMut, s};

/// Convolution/Deconvolution implementation with FFT optimization
/// Supports both convolution (isign=1) and deconvolution (isign=-1)
pub fn convlv(
    data: &[f64],
    respns: &[f64],
    isign: i32,
) -> Result<Array1<f64>, ConvlvError> {
    // Validate inputs
    let n = data.len();
    let m = respns.len();
    
    if n == 0 || m == 0 {
        return Err(ConvlvError::EmptyInput);
    }
    if m > n {
        return Err(ConvlvError::ResponseTooLong);
    }
    if isign != 1 && isign != -1 {
        return Err(ConvlvError::InvalidIsign);
    }

    // Prepare response function with zero-padding
    let mut respns_padded = Array1::zeros(n);
    prepare_response_function(&mut respns_padded, respns, n, m);

    // Compute FFTs in parallel
    let (data_fft, respns_fft) = compute_parallel_ffts(data, &respns_padded, n);

    // Process frequency domain with optimization
    let mut ans_fft = process_frequency_domain(&data_fft, &respns_fft, isign, n)?;

    // Inverse FFT
    realft(&mut ans_fft, n, -1);

    // Normalize and return result
    Ok(ans_fft.slice(s![..n]).to_owned())
}

/// Prepare the response function with zero-padding and proper alignment
fn prepare_response_function(
    respns_padded: &mut Array1<f64>,
    respns: &[f64],
    n: usize,
    m: usize,
) {
    let mid_point = (m - 1) / 2;
    
    // Copy response function with wrap-around for circular convolution
    respns_padded
        .par_chunks_mut(1)
        .enumerate()
        .for_each(|(i, elem)| {
            if i < mid_point {
                // Wrap-around from the end
                *elem = respns[m - mid_point + i];
            } else if i < m {
                // Direct copy
                *elem = respns[i];
            } else if i < n - mid_point {
                // Zero padding
                *elem = 0.0;
            } else {
                // Wrap-around from the beginning
                *elem = respns[i - (n - mid_point)];
            }
        });
}

/// Compute FFTs of data and response function in parallel
fn compute_parallel_ffts(data: &[f64], respns_padded: &Array1<f64>, n: usize) -> (Array1<f64>, Array1<f64>) {
    let (data_fft, respns_fft) = rayon::join(
        || compute_fft(data, n),
        || compute_fft(respns_padded.as_slice().unwrap(), n),
    );
    
    (data_fft, respns_fft)
}

/// Compute FFT with optimization
fn compute_fft(input: &[f64], n: usize) -> Array1<f64> {
    let mut output = Array1::zeros(2 * n);
    output.slice_mut(s![..n]).assign(&Array1::from_vec(input.to_vec()));
    realft(&mut output, n, 1);
    output
}

/// Process frequency domain data with optimization
fn process_frequency_domain(
    data_fft: &Array1<f64>,
    respns_fft: &Array1<f64>,
    isign: i32,
    n: usize,
) -> Result<Array1<f64>, ConvlvError> {
    let no2 = n >> 1;
    let mut ans_fft = Array1::zeros(2 * n);

    // Process complex pairs in parallel
    ans_fft
        .par_chunks_mut(2)
        .zip(data_fft.par_chunks(2))
        .zip(respns_fft.par_chunks(2))
        .enumerate()
        .for_each(|(i, ((ans_chunk, data_chunk), respns_chunk))| {
            if i >= 1 && i < n + 1 {
                let data_re = data_chunk[0];
                let data_im = data_chunk[1];
                let respns_re = respns_chunk[0];
                let respns_im = respns_chunk[1];

                if isign == 1 {
                    // Convolution: multiply in frequency domain
                    ans_chunk[0] = (data_re * respns_re - data_im * respns_im) / no2 as f64;
                    ans_chunk[1] = (data_re * respns_im + data_im * respns_re) / no2 as f64;
                } else {
                    // Deconvolution: divide in frequency domain
                    let mag2 = respns_re * respns_re + respns_im * respns_im;
                    if mag2 < 1e-12 {
                        // Handle near-zero cases gracefully
                        ans_chunk[0] = 0.0;
                        ans_chunk[1] = 0.0;
                    } else {
                        ans_chunk[0] = (data_re * respns_re + data_im * respns_im) / mag2 / no2 as f64;
                        ans_chunk[1] = (data_im * respns_re - data_re * respns_im) / mag2 / no2 as f64;
                    }
                }
            }
        });

    // Handle the Nyquist frequency component
    ans_fft[2 * n - 2] = ans_fft[1];
    ans_fft[2 * n - 1] = 0.0;

    Ok(ans_fft)
}

/// Optimized version using manual vectorization
fn process_frequency_domain_optimized(
    data_fft: &Array1<f64>,
    respns_fft: &Array1<f64>,
    isign: i32,
    n: usize,
) -> Result<Array1<f64>, ConvlvError> {
    let no2 = n >> 1;
    let mut ans_fft = Array1::zeros(2 * n);

    // Process in chunks for better cache performance
    for chunk_start in (1..n).step_by(4) {
        let chunk_end = (chunk_start + 4).min(n);
        
        for i in chunk_start..chunk_end {
            let idx = 2 * i;
            let data_re = data_fft[idx];
            let data_im = data_fft[idx + 1];
            let respns_re = respns_fft[idx];
            let respns_im = respns_fft[idx + 1];

            if isign == 1 {
                // Complex multiplication
                ans_fft[idx] = (data_re * respns_re - data_im * respns_im) / no2 as f64;
                ans_fft[idx + 1] = (data_re * respns_im + data_im * respns_re) / no2 as f64;
            } else {
                // Complex division
                let mag2 = respns_re * respns_re + respns_im * respns_im;
                if mag2 < 1e-12 {
                    ans_fft[idx] = 0.0;
                    ans_fft[idx + 1] = 0.0;
                } else {
                    ans_fft[idx] = (data_re * respns_re + data_im * respns_im) / mag2 / no2 as f64;
                    ans_fft[idx + 1] = (data_im * respns_re - data_re * respns_im) / mag2 / no2 as f64;
                }
            }
        }
    }

    // Handle the Nyquist frequency component
    ans_fft[2 * n - 2] = ans_fft[1];
    ans_fft[2 * n - 1] = 0.0;

    Ok(ans_fft)
}

/// Complex division function (replaces missing complex_divide_simd)
#[inline(always)]
fn complex_divide(numerator_real: f64, numerator_imag: f64, 
                  denominator_real: f64, denominator_imag: f64) -> (f64, f64) {
    let denominator_norm = denominator_real * denominator_real + denominator_imag * denominator_imag;
    if denominator_norm == 0.0 {
        return (0.0, 0.0);
    }
    
    let real = (numerator_real * denominator_real + numerator_imag * denominator_imag) / denominator_norm;
    let imag = (numerator_imag * denominator_real - numerator_real * denominator_imag) / denominator_norm;
    
    (real, imag)
}

/// Complex multiplication function
#[inline(always)]
fn complex_multiply(a_real: f64, a_imag: f64, b_real: f64, b_imag: f64) -> (f64, f64) {
    let real = a_real * b_real - a_imag * b_imag;
    let imag = a_real * b_imag + a_imag * b_real;
    
    (real, imag)
}

/// Wrapper for realft function (using the implementation from Real_FT.rs)
fn realft(data: &mut Array1<f64>, n: usize, isign: i32) {
    // Convert to slice and call the realft function
    let slice = data.as_slice_mut().unwrap();
    crate::Real_FT::realft(slice, n, isign);
}

/// Error types for convolution/deconvolution
#[derive(Debug, thiserror::Error)]
pub enum ConvlvError {
    #[error("Input arrays cannot be empty")]
    EmptyInput,
    #[error("Response function longer than data")]
    ResponseTooLong,
    #[error("isign must be 1 (convolution) or -1 (deconvolution)")]
    InvalidIsign,
    #[error("Division by zero in deconvolution")]
    DivisionByZero,
    #[error("FFT computation error: {0}")]
    FftError(String),
}

/// Batch processing for multiple convolutions
pub fn convlv_batch(
    data_batch: &[&[f64]],
    respns: &[f64],
    isign: i32,
) -> Result<Vec<Array1<f64>>, ConvlvError> {
    data_batch
        .par_iter()
        .map(|data| convlv(data, respns, isign))
        .collect()
}

/// Convolution processor with configurable optimization
pub struct ConvlvProcessor {
    use_optimized: bool,
    parallel_threshold: usize,
}

impl ConvlvProcessor {
    pub fn new() -> Self {
        Self {
            use_optimized: true,
            parallel_threshold: 1024,
        }
    }
    
    pub fn with_optimized(mut self, use_optimized: bool) -> Self {
        self.use_optimized = use_optimized;
        self
    }
    
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }
    
    pub fn process(&self, data: &[f64], respns: &[f64], isign: i32) -> Result<Array1<f64>, ConvlvError> {
        let n = data.len();
        
        if self.use_optimized && n >= self.parallel_threshold {
            self.convlv_optimized(data, respns, isign)
        } else {
            convlv(data, respns, isign)
        }
    }
    
    fn convlv_optimized(&self, data: &[f64], respns: &[f64], isign: i32) -> Result<Array1<f64>, ConvlvError> {
        let n = data.len();
        let m = respns.len();
        
        if n == 0 || m == 0 {
            return Err(ConvlvError::EmptyInput);
        }
        if m > n {
            return Err(ConvlvError::ResponseTooLong);
        }
        if isign != 1 && isign != -1 {
            return Err(ConvlvError::InvalidIsign);
        }

        // Prepare response function with zero-padding
        let mut respns_padded = Array1::zeros(n);
        prepare_response_function(&mut respns_padded, respns, n, m);

        // Compute FFTs
        let data_fft = compute_fft(data, n);
        let respns_fft = compute_fft(respns_padded.as_slice().unwrap(), n);

        // Process frequency domain with optimized version
        let mut ans_fft = process_frequency_domain_optimized(&data_fft, &respns_fft, isign, n)?;

        // Inverse FFT
        realft(&mut ans_fft, n, -1);

        // Normalize and return result
        Ok(ans_fft.slice(s![..n]).to_owned())
    }
    
    pub fn process_batch(&self, data_batch: &[&[f64]], respns: &[f64], isign: i32) -> Result<Vec<Array1<f64>>, ConvlvError> {
        data_batch
            .par_iter()
            .map(|data| self.process(data, respns, isign))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_convolution_basic() -> Result<(), ConvlvError> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let respns = vec![1.0, 1.0];
        
        let result = convlv(&data, &respns, 1)?;
        
        // Expected: [1*1, 1*2+1*1, 1*3+1*2, 1*4+1*3] = [1, 3, 5, 7]
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 7.0, epsilon = 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_deconvolution() -> Result<(), ConvlvError> {
        let data = vec![1.0, 3.0, 5.0, 7.0];
        let respns = vec![1.0, 1.0];
        
        let result = convlv(&data, &respns, -1)?;
        
        // Should recover original [1, 2, 3, 4]
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 4.0, epsilon = 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_complex_divide() {
        let (real, imag) = complex_divide(1.0, 0.0, 1.0, 0.0);
        assert_relative_eq!(real, 1.0, epsilon = 1e-10);
        assert_relative_eq!(imag, 0.0, epsilon = 1e-10);
        
        let (real, imag) = complex_divide(1.0, 1.0, 1.0, 1.0);
        assert_relative_eq!(real, 1.0, epsilon = 1e-10);
        assert_relative_eq!(imag, 0.0, epsilon = 1e-10);
        
        // Test division by zero handling
        let (real, imag) = complex_divide(1.0, 1.0, 0.0, 0.0);
        assert_relative_eq!(real, 0.0, epsilon = 1e-10);
        assert_relative_eq!(imag, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_multiply() {
        let (real, imag) = complex_multiply(1.0, 0.0, 1.0, 0.0);
        assert_relative_eq!(real, 1.0, epsilon = 1e-10);
        assert_relative_eq!(imag, 0.0, epsilon = 1e-10);
        
        let (real, imag) = complex_multiply(1.0, 1.0, 1.0, 1.0);
        assert_relative_eq!(real, 0.0, epsilon = 1e-10); // (1+1i)*(1+1i) = 0+2i
        assert_relative_eq!(imag, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_error_handling() {
        // Test empty input
        assert!(matches!(convlv(&[], &[1.0], 1), Err(ConvlvError::EmptyInput)));
        
        // Test response too long
        assert!(matches!(
            convlv(&[1.0, 2.0], &[1.0, 2.0, 3.0], 1),
            Err(ConvlvError::ResponseTooLong)
        ));
        
        // Test invalid isign
        assert!(matches!(
            convlv(&[1.0, 2.0], &[1.0], 0),
            Err(ConvlvError::InvalidIsign)
        ));
    }

    #[test]
    fn test_batch_processing() -> Result<(), ConvlvError> {
        let batch = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let respns = vec![1.0, 1.0];
        
        let results = convlv_batch(&batch.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &respns, 1)?;
        
        assert_eq!(results.len(), 2);
        assert_relative_eq!(results[0][1], 3.0, epsilon = 1e-10); // 1+2
        assert_relative_eq!(results[1][1], 9.0, epsilon = 1e-10); // 4+5
        
        Ok(())
    }

    #[test]
    fn test_processor() -> Result<(), ConvlvError> {
        let processor = ConvlvProcessor::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let respns = vec![1.0, 1.0];
        
        let result = processor.process(&data, &respns, 1)?;
        
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        Ok(())
    }
}
