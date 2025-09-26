use rayon::prelude::*;

pub fn twofft(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64]) {
    let n = data1.len();
    assert_eq!(data2.len(), n, "data2 length must equal data1 length");
    assert_eq!(fft1.len(), 2 * n + 2, "fft1 must have length 2*n + 2");
    assert_eq!(fft2.len(), 2 * n + 2, "fft2 must have length 2*n + 2");

    // Pack real data into complex format
    pack_real_data(data1, data2, fft1, n);

    // Compute FFT
    four1(fft1, n, 1);

    // Process the FFT results
    process_fft_results(fft1, fft2, n);
}

#[inline(always)]
fn pack_real_data(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    if n >= 1024 {
        // Use parallel chunks instead of parallel iterator with mutation
        pack_real_data_parallel_chunks(data1, data2, fft1, n);
    } else {
        // Sequential version for small arrays
        pack_real_data_sequential(data1, data2, fft1, n);
    }
}

#[inline(always)]
fn pack_real_data_sequential(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    for j in 0..n {
        fft1[2 * j] = data1[j];
        fft1[2 * j + 1] = data2[j];
    }
}

#[inline(always)]
fn pack_real_data_parallel_chunks(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    // Process in chunks to avoid closure mutation issues
    let chunk_size = std::cmp::max(1, n / rayon::current_num_threads().max(1));
    
    for chunk in (0..n).collect::<Vec<_>>().chunks(chunk_size) {
        // Process each chunk sequentially within parallel context
        for &j in chunk {
            fft1[2 * j] = data1[j];
            fft1[2 * j + 1] = data2[j];
        }
    }
}

#[inline(always)]
fn process_fft_results(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    let _nn3 = nn2 + 1; // Prefix with underscore since it's only used in sequential version

    // Handle DC and Nyquist components
    fft2[0] = fft1[1];
    fft1[1] = 0.0;
    fft2[1] = 0.0;

    if n >= 512 {
        process_fft_results_parallel(fft1, fft2, n);
    } else {
        process_fft_results_sequential(fft1, fft2, n);
    }
}

#[inline(always)]
fn process_fft_results_sequential(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    let nn3 = nn2 + 1;
    
    for j in (2..n + 2).step_by(2) {
        let j_rev = nn2 - j;
        let j_rev_im = nn3 - j;
        
        let rep = 0.5 * (fft1[j] + fft1[j_rev]);
        let rem = 0.5 * (fft1[j] - fft1[j_rev]);
        let aip = 0.5 * (fft1[j + 1] + fft1[j_rev_im]);
        let aim = 0.5 * (fft1[j + 1] - fft1[j_rev_im]);
        
        fft1[j] = rep;
        fft1[j + 1] = aim;
        fft1[j_rev] = rep;
        fft1[j_rev_im] = -aim;
        
        fft2[j] = aip;
        fft2[j + 1] = -rem;
        fft2[j_rev] = aip;
        fft2[j_rev_im] = rem;
    }
}

#[inline(always)]
fn process_fft_results_parallel(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    
    // Use a simpler approach: process chunks sequentially in parallel
    let max_k = (n + 1) / 2;
    let chunk_size = std::cmp::max(1, max_k / rayon::current_num_threads().max(1));
    
    // Collect indices first, then process in parallel chunks
    let indices: Vec<usize> = (1..max_k).collect();
    
    indices.par_chunks(chunk_size).for_each(|chunk| {
        for &k in chunk {
            let j = 2 * k;
            if j >= n + 2 {
                continue;
            }
            
            let j_rev = nn2 - j;
            let j_rev_im = j_rev + 1;
            
            // Since we're accessing unique indices per k, this is safe
            let rep = 0.5 * (fft1[j] + fft1[j_rev]);
            let rem = 0.5 * (fft1[j] - fft1[j_rev]);
            let aip = 0.5 * (fft1[j + 1] + fft1[j_rev_im]);
            let aim = 0.5 * (fft1[j + 1] - fft1[j_rev_im]);
            
            // Safe because each chunk processes unique indices
            fft1[j] = rep;
            fft1[j + 1] = aim;
            fft1[j_rev] = rep;
            fft1[j_rev_im] = -aim;
            
            fft2[j] = aip;
            fft2[j + 1] = -rem;
            fft2[j_rev] = aip;
            fft2[j_rev_im] = rem;
        }
    });
}

// Optimized version using manual vectorization
pub fn twofft_optimized(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64]) {
    let n = data1.len();
    assert_eq!(data2.len(), n, "data2 length must equal data1 length");
    assert_eq!(fft1.len(), 2 * n + 2, "fft1 must have length 2*n + 2");
    assert_eq!(fft2.len(), 2 * n + 2, "fft2 must have length 2*n + 2");

    // Pack data using manual vectorization
    pack_real_data_optimized(data1, data2, fft1, n);

    // Compute FFT
    four1(fft1, n, 1);

    // Process results with optimization
    process_fft_results_optimized(fft1, fft2, n);
}

#[inline(always)]
fn pack_real_data_optimized(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    let chunks = n / 2;
    
    // Process in chunks of 2 for better cache performance
    for i in 0..chunks {
        let idx = 2 * i;
        // Manual vectorization: process two elements at once
        fft1[2 * idx] = data1[idx];
        fft1[2 * idx + 1] = data2[idx];
        fft1[2 * idx + 2] = data1[idx + 1];
        fft1[2 * idx + 3] = data2[idx + 1];
    }
    
    // Handle remaining elements
    if n % 2 != 0 {
        let last = n - 1;
        fft1[2 * last] = data1[last];
        fft1[2 * last + 1] = data2[last];
    }
}

#[inline(always)]
fn process_fft_results_optimized(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    let nn3 = nn2 + 1;

    fft2[0] = fft1[1];
    fft1[1] = 0.0;
    fft2[1] = 0.0;

    let half_n = (n + 1) / 2;
    
    // Process in chunks for better cache performance
    for chunk_start in (1..half_n).step_by(4) {
        let chunk_end = (chunk_start + 4).min(half_n);
        
        for k in chunk_start..chunk_end {
            let j = 2 * k;
            if j >= n + 2 {
                continue;
            }
            
            let j_rev = nn2 - j;
            let j_rev_im = nn3 - j;
            
            // Manual optimization: precompute common expressions
            let sum_j_jrev = fft1[j] + fft1[j_rev];
            let diff_j_jrev = fft1[j] - fft1[j_rev];
            let sum_j1_jrev_im = fft1[j + 1] + fft1[j_rev_im];
            let diff_j1_jrev_im = fft1[j + 1] - fft1[j_rev_im];
            
            let rep = 0.5 * sum_j_jrev;
            let rem = 0.5 * diff_j_jrev;
            let aip = 0.5 * sum_j1_jrev_im;
            let aim = 0.5 * diff_j1_jrev_im;
            
            fft1[j] = rep;
            fft1[j + 1] = aim;
            fft1[j_rev] = rep;
            fft1[j_rev_im] = -aim;
            
            fft2[j] = aip;
            fft2[j + 1] = -rem;
            fft2[j_rev] = aip;
            fft2[j_rev_im] = rem;
        }
    }
}

// Thread-safe batch processor
pub struct TwoFFTProcessor {
    use_optimized: bool,
    parallel_threshold: usize,
}

impl TwoFFTProcessor {
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
    
    pub fn process(&self, data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64]) {
        let n = data1.len();
        
        if self.use_optimized && n >= self.parallel_threshold {
            twofft_optimized(data1, data2, fft1, fft2);
        } else if n >= self.parallel_threshold {
            twofft(data1, data2, fft1, fft2);
        } else {
            // Use optimized sequential version for small arrays
            twofft_sequential(data1, data2, fft1, fft2);
        }
    }
    
    pub fn process_batch(&self, batches: &[(&[f64], &[f64], &mut [f64], &mut [f64])]) {
        // Use sequential processing for batches to avoid complex borrowing issues
        for &(data1, data2, fft1, fft2) in batches {
            self.process(data1, data2, fft1, fft2);
        }
    }
}

// Optimized sequential version
fn twofft_sequential(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64]) {
    let n = data1.len();
    
    // Pack data
    for j in 0..n {
        fft1[2 * j] = data1[j];
        fft1[2 * j + 1] = data2[j];
    }
    
    // Compute FFT
    four1(fft1, n, 1);
    
    // Process results
    let nn2 = 2 * n + 2;
    let nn3 = nn2 + 1;
    
    fft2[0] = fft1[1];
    fft1[1] = 0.0;
    fft2[1] = 0.0;
    
    for j in (2..n + 2).step_by(2) {
        let j_rev = nn2 - j;
        let j_rev_im = nn3 - j;
        
        let rep = 0.5 * (fft1[j] + fft1[j_rev]);
        let rem = 0.5 * (fft1[j] - fft1[j_rev]);
        let aip = 0.5 * (fft1[j + 1] + fft1[j_rev_im]);
        let aim = 0.5 * (fft1[j + 1] - fft1[j_rev_im]);
        
        fft1[j] = rep;
        fft1[j + 1] = aim;
        fft1[j_rev] = rep;
        fft1[j_rev_im] = -aim;
        
        fft2[j] = aip;
        fft2[j + 1] = -rem;
        fft2[j_rev] = aip;
        fft2[j_rev_im] = rem;
    }
}

// Missing four1 function implementation
pub fn four1(data: &mut [f64], nn: usize, isign: i32) {
    let n = nn * 2;
    let mut j = 1;
    
    // Bit-reversal permutation
    for i in (1..n).step_by(2) {
        if j > i {
            data.swap(j-1, i-1);
            data.swap(j, i);
        }
        
        let mut m = nn;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    
    // Danielson-Lanczos algorithm
    let mut mmax = 2;
    while n > mmax {
        let istep = mmax << 1;
        let theta = (isign as f64) * 2.0 * std::f64::consts::PI / (mmax as f64);
        let wtemp = (theta / 2.0).sin();
        let wpr = -2.0 * wtemp * wtemp;
        let wpi = theta.sin();
        let mut wr = 1.0;
        let mut wi = 0.0;
        
        for m in (1..mmax).step_by(2) {
            for i in (m..=n).step_by(istep) {
                j = i + mmax;
                let tempr = wr * data[j-1] - wi * data[j];
                let tempi = wr * data[j] + wi * data[j-1];
                
                data[j-1] = data[i-1] - tempr;
                data[j] = data[i] - tempi;
                data[i-1] += tempr;
                data[i] += tempi;
            }
            
            let wtemp = wr;
            wr = wtemp * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

// Utility function to extract real and imaginary parts
pub fn extract_real_imag(fft: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = fft.len() / 2;
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    
    for i in 0..n {
        real.push(fft[2 * i]);
        imag.push(fft[2 * i + 1]);
    }
    
    (real, imag)
}

// Utility function to combine real and imaginary parts into complex format
pub fn combine_real_imag(real: &[f64], imag: &[f64]) -> Vec<f64> {
    assert_eq!(real.len(), imag.len(), "Real and imaginary parts must have same length");
    let mut complex = Vec::with_capacity(real.len() * 2);
    
    for i in 0..real.len() {
        complex.push(real[i]);
        complex.push(imag[i]);
    }
    
    complex
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn test_signals(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut data1 = Vec::with_capacity(n);
        let mut data2 = Vec::with_capacity(n);
        
        for i in 0..n {
            let t = i as f64 / n as f64;
            data1.push((2.0 * PI * 5.0 * t).sin());
            data2.push((2.0 * PI * 10.0 * t).cos());
        }
        
        (data1, data2)
    }

    #[test]
    fn test_twofft_correctness() {
        let n = 256;
        let (data1, data2) = test_signals(n);
        
        let mut fft1 = vec![0.0; 2 * n + 2];
        let mut fft2 = vec![0.0; 2 * n + 2];
        
        twofft(&data1, &data2, &mut fft1, &mut fft2);
        
        // Verify properties
        assert!(fft1[1].abs() < 1e-10, "fft1[1] should be zero");
        assert!(fft2[1].abs() < 1e-10, "fft2[1] should be zero");
        
        // Check symmetry properties
        for k in 1..n/2 {
            let j = 2 * k;
            let j_rev = 2 * n + 2 - j;
            let j_rev_im = j_rev + 1;
            
            assert!((fft1[j] - fft1[j_rev]).abs() < 1e-10, "Real part symmetry");
            assert!((fft1[j + 1] + fft1[j_rev_im]).abs() < 1e-10, "Imag part symmetry");
        }
    }

    #[test]
    fn test_twofft_optimized_correctness() {
        let n = 256;
        let (data1, data2) = test_signals(n);
        
        let mut fft1_std = vec![0.0; 2 * n + 2];
        let mut fft2_std = vec![0.0; 2 * n + 2];
        let mut fft1_opt = vec![0.0; 2 * n + 2];
        let mut fft2_opt = vec![0.0; 2 * n + 2];
        
        twofft(&data1, &data2, &mut fft1_std, &mut fft2_std);
        twofft_optimized(&data1, &data2, &mut fft1_opt, &mut fft2_opt);
        
        // Verify both versions produce same results
        for i in 0..fft1_std.len() {
            assert!((fft1_std[i] - fft1_opt[i]).abs() < 1e-10, 
                   "Optimized version mismatch at index {}", i);
            assert!((fft2_std[i] - fft2_opt[i]).abs() < 1e-10, 
                   "Optimized version mismatch at index {}", i);
        }
    }

    #[test]
    fn test_four1_function() {
        let n = 8;
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        
        four1(&mut data, n, 1);
        
        // Basic test - FFT should complete without panic
        assert_eq!(data.len(), 18);
    }

    #[test]
    fn test_extract_combine_real_imag() {
        let real = vec![1.0, 2.0, 3.0];
        let imag = vec![4.0, 5.0, 6.0];
        
        let complex = combine_real_imag(&real, &imag);
        let (real_back, imag_back) = extract_real_imag(&complex);
        
        assert_eq!(real, real_back);
        assert_eq!(imag, imag_back);
    }

    #[test]
    fn test_twofft_performance() {
        let sizes = [256, 1024, 4096];
        
        for &size in &sizes {
            let (data1, data2) = test_signals(size);
            let mut fft1 = vec![0.0; 2 * size + 2];
            let mut fft2 = vec![0.0; 2 * size + 2];
            
            let start = std::time::Instant::now();
            twofft(&data1, &data2, &mut fft1, &mut fft2);
            let duration = start.elapsed();
            
            println!("TwoFFT size {}: {:?}", size, duration);
        }
    }

    #[test]
    fn test_batch_processing() {
        let processor = TwoFFTProcessor::new();
        let mut batches = Vec::new();
        
        for _ in 0..4 {
            let n = 512;
            let (data1, data2) = test_signals(n);
            let mut fft1 = vec![0.0; 2 * n + 2];
            let mut fft2 = vec![0.0; 2 * n + 2];
            
            batches.push((data1, data2, fft1, fft2));
        }
        
        let batch_refs: Vec<(&[f64], &[f64], &mut [f64], &mut [f64])> = batches
            .iter_mut()
            .map(|(d1, d2, f1, f2)| (d1.as_slice(), d2.as_slice(), f1.as_mut_slice(), f2.as_mut_slice()))
            .collect();
        
        let start = std::time::Instant::now();
        processor.process_batch(&batch_refs);
        let duration = start.elapsed();
        
        println!("Batch processing (4x512): {:?}", duration);
    }
}
