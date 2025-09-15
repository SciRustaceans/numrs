use rayon::prelude::*;
use std::simd::{f64x2, Simd, SimdFloat};

pub fn twofft(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    assert_eq!(data1.len(), n, "data1 length must equal n");
    assert_eq!(data2.len(), n, "data2 length must equal n");
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
        // Parallel version for large arrays
        pack_real_data_parallel(data1, data2, fft1, n);
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
fn pack_real_data_parallel(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    fft1.par_chunks_mut(2)
        .zip(data1.par_iter())
        .zip(data2.par_iter())
        .for_each(|((chunk, &d1), &d2)| {
            chunk[0] = d1;
            chunk[1] = d2;
        });
}

#[inline(always)]
fn process_fft_results(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    let nn3 = nn2 + 1;

    // Handle DC and Nyquist components
    fft2[0] = fft1[1];  // fft2[1] in 1-indexed
    fft1[1] = 0.0;
    fft2[1] = 0.0;

    if n >= 512 {
        process_fft_results_parallel(fft1, fft2, n, nn2, nn3);
    } else {
        process_fft_results_sequential(fft1, fft2, n, nn2, nn3);
    }
}

#[inline(always)]
fn process_fft_results_sequential(fft1: &mut [f64], fft2: &mut [f64], n: usize, nn2: usize, nn3: usize) {
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
fn process_fft_results_parallel(fft1: &mut [f64], fft2: &mut [f64], n: usize, nn2: usize, nn3: usize) {
    let half_n = (n + 1) / 2;
    
    (1..half_n).into_par_iter().for_each(|k| {
        let j = 2 * k;
        if j >= n + 2 {
            return;
        }
        
        let j_rev = nn2 - j;
        let j_rev_im = nn3 - j;
        
        // Use SIMD for parallel computation if available
        #[cfg(feature = "simd")]
        {
            let fft1_j = f64x2::from_slice(&fft1[j..j+2]);
            let fft1_rev = f64x2::from_slice(&fft1[j_rev..j_rev+2]);
            
            let sum = fft1_j + fft1_rev;
            let diff = fft1_j - fft1_rev;
            
            let rep = 0.5 * sum[0];
            let rem = 0.5 * diff[0];
            let aip = 0.5 * sum[1];
            let aim = 0.5 * diff[1];
            
            fft1[j] = rep;
            fft1[j + 1] = aim;
            fft1[j_rev] = rep;
            fft1[j_rev_im] = -aim;
            
            fft2[j] = aip;
            fft2[j + 1] = -rem;
            fft2[j_rev] = aip;
            fft2[j_rev_im] = rem;
        }
        
        #[cfg(not(feature = "simd"))]
        {
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
    });
}

// SIMD-optimized version
pub fn twofft_simd(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    assert_eq!(data1.len(), n, "data1 length must equal n");
    assert_eq!(data2.len(), n, "data2 length must equal n");
    assert_eq!(fft1.len(), 2 * n + 2, "fft1 must have length 2*n + 2");
    assert_eq!(fft2.len(), 2 * n + 2, "fft2 must have length 2*n + 2");

    // Pack data using SIMD
    pack_real_data_simd(data1, data2, fft1, n);

    // Compute FFT
    four1(fft1, n, 1);

    // Process results with SIMD
    process_fft_results_simd(fft1, fft2, n);
}

#[inline(always)]
fn pack_real_data_simd(data1: &[f64], data2: &[f64], fft1: &mut [f64], n: usize) {
    let chunks = n / 2;
    
    for i in 0..chunks {
        let idx = 2 * i;
        let data1_vec = f64x2::from_slice(&data1[idx..idx+2]);
        let data2_vec = f64x2::from_slice(&data2[idx..idx+2]);
        
        // Interleave real and imaginary parts
        let interleaved = Simd::swizzle_dyn(
            Simd::concat(data1_vec, data2_vec),
            [0, 2, 1, 3]
        );
        
        interleaved.copy_to_slice(&mut fft1[2*idx..2*idx+4]);
    }
    
    // Handle remaining elements
    if n % 2 != 0 {
        let last = n - 1;
        fft1[2 * last] = data1[last];
        fft1[2 * last + 1] = data2[last];
    }
}

#[inline(always)]
fn process_fft_results_simd(fft1: &mut [f64], fft2: &mut [f64], n: usize) {
    let nn2 = 2 * n + 2;
    let nn3 = nn2 + 1;

    fft2[0] = fft1[1];
    fft1[1] = 0.0;
    fft2[1] = 0.0;

    let half_n = (n + 1) / 2;
    
    for k in 1..half_n {
        let j = 2 * k;
        if j >= n + 2 {
            continue;
        }
        
        let j_rev = nn2 - j;
        let j_rev_im = nn3 - j;
        
        // Load both complex numbers at once
        let current = f64x2::from_slice(&fft1[j..j+2]);
        let reversed = f64x2::from_slice(&fft1[j_rev..j_rev+2]);
        
        let sum = current + reversed;
        let diff = current - reversed;
        
        let half = f64x2::splat(0.5);
        let rep_aim = half * f64x2::from_array([sum[0], diff[1]]);
        let aip_rem = half * f64x2::from_array([sum[1], -diff[0]]);
        
        // Store results
        fft1[j] = rep_aim[0];
        fft1[j + 1] = rep_aim[1];
        fft1[j_rev] = rep_aim[0];
        fft1[j_rev_im] = -rep_aim[1];
        
        fft2[j] = aip_rem[0];
        fft2[j + 1] = aip_rem[1];
        fft2[j_rev] = aip_rem[0];
        fft2[j_rev_im] = -aip_rem[1];
    }
}

// Thread-safe batch processor
pub struct TwoFFTProcessor {
    use_simd: bool,
    parallel_threshold: usize,
}

impl TwoFFTProcessor {
    pub fn new() -> Self {
        Self {
            use_simd: true,
            parallel_threshold: 1024,
        }
    }
    
    pub fn with_simd(mut self, use_simd: bool) -> Self {
        self.use_simd = use_simd;
        self
    }
    
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }
    
    pub fn process(&self, data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64]) {
        let n = data1.len();
        
        if self.use_simd && n >= self.parallel_threshold {
            twofft_simd(data1, data2, fft1, fft2, n);
        } else if n >= self.parallel_threshold {
            twofft(data1, data2, fft1, fft2, n);
        } else {
            // Use optimized sequential version for small arrays
            twofft_sequential(data1, data2, fft1, fft2, n);
        }
    }
    
    pub fn process_batch(&self, batches: &[(&[f64], &[f64], &mut [f64], &mut [f64])]) {
        batches.par_iter().for_each(|(data1, data2, fft1, fft2)| {
            self.process(data1, data2, fft1, fft2);
        });
    }
}

// Optimized sequential version
fn twofft_sequential(data1: &[f64], data2: &[f64], fft1: &mut [f64], fft2: &mut [f64], n: usize) {
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
        
        twofft(&data1, &data2, &mut fft1, &mut fft2, n);
        
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
    fn test_twofft_performance() {
        let sizes = [256, 1024, 4096, 16384];
        
        for &size in &sizes {
            let (data1, data2) = test_signals(size);
            let mut fft1 = vec![0.0; 2 * size + 2];
            let mut fft2 = vec![0.0; 2 * size + 2];
            
            let start = std::time::Instant::now();
            twofft(&data1, &data2, &mut fft1, &mut fft2, size);
            let duration = start.elapsed();
            
            println!("TwoFFT size {}: {:?}", size, duration);
        }
    }

    #[test]
    fn test_batch_processing() {
        let processor = TwoFFTProcessor::new();
        let mut batches = Vec::new();
        
        for i in 0..4 {
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
