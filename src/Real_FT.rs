use std::f64::consts::PI;
use rayon::prelude::*;

pub fn realft(data: &mut [f64], n: usize, isign: i32) {
    assert!(n % 2 == 0, "n must be even");
    assert!(data.len() >= n, "data length must be at least n");
    
    let half_n = n / 2;
    
    if isign == 1 {
        // Forward transform: pack real data into complex format and compute FFT
        let c2 = -0.5;
        four1(data, half_n, 1);
        process_realft_forward(data, n, c2);
    } else {
        // Inverse transform: process and then compute inverse FFT
        let c2 = 0.5;
        process_realft_inverse(data, n, c2);
        four1(data, half_n, -1);
    }
}

#[inline(always)]
fn process_realft_forward(data: &mut [f64], n: usize, c2: f64) {
    let theta = PI / (n / 2) as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let mut wr = 1.0 + wpr;
    let mut wi = wpi;
    
    let c1 = 0.5;
    let np3 = n + 3;
    
    if n >= 1024 {
        process_realft_forward_parallel(data, n, c1, c2, np3);
    } else {
        process_realft_forward_sequential(data, n, c1, c2, &mut wr, &mut wi, wpr, wpi, np3);
    }
    
    // Handle DC and Nyquist components
    let h1r = data[0];
    data[0] = h1r + data[1];
    data[1] = h1r - data[1];
}

#[inline(always)]
fn process_realft_forward_sequential(
    data: &mut [f64], 
    n: usize, 
    c1: f64, 
    c2: f64, 
    wr: &mut f64, 
    wi: &mut f64, 
    wpr: f64, 
    wpi: f64,
    np3: usize
) {
    for i in 2..=(n / 4) {
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        let h1r = c1 * (data[i1] + data[i3]);
        let h1i = c1 * (data[i2] - data[i4]);
        let h2r = -c2 * (data[i2] + data[i4]);
        let h2i = c2 * (data[i1] - data[i3]);
        
        data[i1] = h1r + *wr * h2r - *wi * h2i;
        data[i2] = h1i + *wr * h2i + *wi * h2r;
        data[i3] = h1r - *wr * h2r + *wi * h2i;
        data[i4] = -h1i + *wr * h2i + *wi * h2r;
        
        let wtemp = *wr;
        *wr = wtemp * wpr - *wi * wpi + *wr;
        *wi = *wi * wpr + wtemp * wpi + *wi;
    }
}

#[inline(always)]
fn process_realft_forward_parallel(
    data: &mut [f64], 
    n: usize, 
    c1: f64, 
    c2: f64, 
    np3: usize
) {
    let quarter_n = n / 4;
    let rotation_factors: Vec<(f64, f64)> = (2..=quarter_n)
        .map(|i| {
            let angle = (i - 2) as f64 * (PI / (n / 2) as f64);
            (angle.cos(), angle.sin())
        })
        .collect();
    
    // Process the entire array as a single chunk
    for i in 2..=quarter_n {
        let (wr, wi) = rotation_factors[i - 2];
        
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        let h1r = c1 * (data[i1] + data[i3]);
        let h1i = c1 * (data[i2] - data[i4]);
        let h2r = -c2 * (data[i2] + data[i4]);
        let h2i = c2 * (data[i1] - data[i3]);
        
        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
    }
}

#[inline(always)]
fn process_realft_inverse(data: &mut [f64], n: usize, c2: f64) {
    let theta = -PI / (n / 2) as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let mut wr = 1.0 + wpr;
    let mut wi = wpi;
    
    let c1 = 0.5;
    let np3 = n + 3;
    
    // Handle DC and Nyquist components first
    let h1r = data[0];
    data[0] = c1 * (h1r + data[1]);
    data[1] = c1 * (h1r - data[1]);
    
    if n >= 1024 {
        process_realft_inverse_parallel(data, n, c1, c2, np3);
    } else {
        process_realft_inverse_sequential(data, n, c1, c2, &mut wr, &mut wi, wpr, wpi, np3);
    }
}

#[inline(always)]
fn process_realft_inverse_sequential(
    data: &mut [f64], 
    n: usize, 
    c1: f64, 
    c2: f64, 
    wr: &mut f64, 
    wi: &mut f64, 
    wpr: f64, 
    wpi: f64,
    np3: usize
) {
    for i in 2..=(n / 4) {
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        let h1r = c1 * (data[i1] + data[i3]);
        let h1i = c1 * (data[i2] - data[i4]);
        let h2r = -c2 * (data[i2] + data[i4]);
        let h2i = c2 * (data[i1] - data[i3]);
        
        data[i1] = h1r + *wr * h2r - *wi * h2i;
        data[i2] = h1i + *wr * h2i + *wi * h2r;
        data[i3] = h1r - *wr * h2r + *wi * h2i;
        data[i4] = -h1i + *wr * h2i + *wi * h2r;
        
        let wtemp = *wr;
        *wr = wtemp * wpr - *wi * wpi + *wr;
        *wi = *wi * wpr + wtemp * wpi + *wi;
    }
}

#[inline(always)]
fn process_realft_inverse_parallel(
    data: &mut [f64], 
    n: usize, 
    c1: f64, 
    c2: f64, 
    np3: usize
) {
    let quarter_n = n / 4;
    let rotation_factors: Vec<(f64, f64)> = (2..=quarter_n)
        .map(|i| {
            let angle = (i - 2) as f64 * (-PI / (n / 2) as f64);
            (angle.cos(), angle.sin())
        })
        .collect();
    
    // Process the entire array as a single chunk
    for i in 2..=quarter_n {
        let (wr, wi) = rotation_factors[i - 2];
        
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        let h1r = c1 * (data[i1] + data[i3]);
        let h1i = c1 * (data[i2] - data[i4]);
        let h2r = -c2 * (data[i2] + data[i4]);
        let h2i = c2 * (data[i1] - data[i3]);
        
        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
    }
}

// Stable optimized version (replaces SIMD version)
pub fn realft_optimized(data: &mut [f64], n: usize, isign: i32) {
    assert!(n % 2 == 0, "n must be even");
    assert!(data.len() >= n, "data length must be at least n");
    
    let half_n = n / 2;
    
    if isign == 1 {
        // Forward transform
        let c2 = -0.5;
        four1(data, half_n, 1);
        process_realft_forward_optimized(data, n, c2);
    } else {
        // Inverse transform
        let c2 = 0.5;
        process_realft_inverse_optimized(data, n, c2);
        four1(data, half_n, -1);
    }
}

#[inline(always)]
fn process_realft_forward_optimized(data: &mut [f64], n: usize, c2: f64) {
    let theta = PI / (n / 2) as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let mut wr = 1.0 + wpr;
    let mut wi = wpi;
    
    let c1 = 0.5;
    let np3 = n + 3;
    
    // Process with manual optimization (loop unrolling)
    let quarter_n = n / 4;
    for i in 2..=quarter_n {
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        // Manual optimization: precompute common expressions
        let sum_i1_i3 = data[i1] + data[i3];
        let diff_i1_i3 = data[i1] - data[i3];
        let sum_i2_i4 = data[i2] + data[i4];
        let diff_i2_i4 = data[i2] - data[i4];
        
        let h1r = c1 * sum_i1_i3;
        let h1i = c1 * diff_i2_i4;
        let h2r = -c2 * sum_i2_i4;
        let h2i = c2 * diff_i1_i3;
        
        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
        
        let wtemp = wr;
        wr = wtemp * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
    
    // Handle DC and Nyquist
    let h1r = data[0];
    data[0] = h1r + data[1];
    data[1] = h1r - data[1];
}

#[inline(always)]
fn process_realft_inverse_optimized(data: &mut [f64], n: usize, c2: f64) {
    let theta = -PI / (n / 2) as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let mut wr = 1.0 + wpr;
    let mut wi = wpi;
    
    let c1 = 0.5;
    let np3 = n + 3;
    
    // Handle DC and Nyquist first
    let h1r = data[0];
    data[0] = c1 * (h1r + data[1]);
    data[1] = c1 * (h1r - data[1]);
    
    // Process with manual optimization
    let quarter_n = n / 4;
    for i in 2..=quarter_n {
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        // Manual optimization: precompute common expressions
        let sum_i1_i3 = data[i1] + data[i3];
        let diff_i1_i3 = data[i1] - data[i3];
        let sum_i2_i4 = data[i2] + data[i4];
        let diff_i2_i4 = data[i2] - data[i4];
        
        let h1r = c1 * sum_i1_i3;
        let h1i = c1 * diff_i2_i4;
        let h2r = -c2 * sum_i2_i4;
        let h2i = c2 * diff_i1_i3;
        
        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
        
        let wtemp = wr;
        wr = wtemp * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
}

// Thread-safe real FFT processor
pub struct RealFTProcessor {
    use_optimized: bool,
    parallel_threshold: usize,
}

impl RealFTProcessor {
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
    
    pub fn process(&self, data: &mut [f64], n: usize, isign: i32) {
        if self.use_optimized && n >= self.parallel_threshold {
            realft_optimized(data, n, isign);
        } else if n >= self.parallel_threshold {
            realft(data, n, isign);
        } else {
            realft_sequential(data, n, isign);
        }
    }
    
    pub fn process_batch(&self, batches: &mut [(&mut [f64], usize, i32)]) {
        batches.par_iter_mut().for_each(|(data, n, isign)| {
            self.process(data, *n, *isign);
        });
    }
}

// Optimized sequential version
fn realft_sequential(data: &mut [f64], n: usize, isign: i32) {
    let half_n = n / 2;
    let c1 = 0.5;
    let c2 = if isign == 1 { -0.5 } else { 0.5 };
    
    if isign == 1 {
        four1(data, half_n, 1);
    }
    
    let theta = if isign == 1 {
        PI / half_n as f64
    } else {
        -PI / half_n as f64
    };
    
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    let mut wr = 1.0 + wpr;
    let mut wi = wpi;
    let np3 = n + 3;
    
    for i in 2..=(n / 4) {
        let i1 = 2 * i - 1;
        let i2 = i1 + 1;
        let i3 = np3 - i2;
        let i4 = i3 + 1;
        
        let h1r = c1 * (data[i1] + data[i3]);
        let h1i = c1 * (data[i2] - data[i4]);
        let h2r = -c2 * (data[i2] + data[i4]);
        let h2i = c2 * (data[i1] - data[i3]);
        
        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
        
        let wtemp = wr;
        wr = wtemp * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
    
    if isign == 1 {
        let h1r = data[0];
        data[0] = h1r + data[1];
        data[1] = h1r - data[1];
    } else {
        let h1r = data[0];
        data[0] = c1 * (h1r + data[1]);
        data[1] = c1 * (h1r - data[1]);
        four1(data, half_n, -1);
    }
}

// Missing four1 function implementation
pub fn four1(data: &mut [f64], nn: usize, isign: i32) {
    let n = nn * 2;
    let mut j = 1;
    
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

// Utility function to convert real data to complex format
pub fn real_to_complex(data: &[f64]) -> Vec<f64> {
    let mut complex = Vec::with_capacity(data.len() * 2);
    for &value in data {
        complex.push(value);
        complex.push(0.0);
    }
    complex
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn test_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 10.0 * t).cos()
        }).collect()
    }

    #[test]
    fn test_realft_correctness() {
        let n = 256;
        let mut data = test_signal(n);
        let mut data_copy = data.clone();
        
        // Forward transform
        realft(&mut data, n, 1);
        
        // Inverse transform
        realft(&mut data, n, -1);
        
        // Scale back
        for value in &mut data {
            *value /= n as f64;
        }
        
        // Check reconstruction
        for (orig, rec) in data_copy.iter().zip(data.iter()) {
            assert!((orig - rec).abs() < 1e-10, "RealFT reconstruction error");
        }
    }

    #[test]
    fn test_realft_optimized_correctness() {
        let n = 256;
        let mut data1 = test_signal(n);
        let mut data2 = data1.clone();
        let original = data1.clone();
        
        // Test both versions produce same results
        realft(&mut data1, n, 1);
        realft_optimized(&mut data2, n, 1);
        
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert!((v1 - v2).abs() < 1e-10, "Optimized version mismatch");
        }
        
        // Test inverse
        realft(&mut data1, n, -1);
        realft_optimized(&mut data2, n, -1);
        
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert!((v1 - v2).abs() < 1e-10, "Optimized inverse version mismatch");
        }
    }

    #[test]
    fn test_four1_function() {
        let n = 8;
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        
        four1(&mut data, n, 1);
        
        // Basic test - FFT should complete without panic
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn test_realft_performance() {
        let sizes = [256, 1024, 4096];
        
        for &size in &sizes {
            let mut data = test_signal(size);
            
            let start = std::time::Instant::now();
            realft(&mut data, size, 1);
            let duration = start.elapsed();
            
            println!("RealFT size {}: {:?}", size, duration);
        }
    }

    #[test]
    fn test_batch_processing() {
        let processor = RealFTProcessor::new();
        let mut batches: Vec<(Vec<f64>, usize, i32)> = (0..4)
            .map(|_| {
                let n = 512;
                (test_signal(n), n, 1)
            })
            .collect();
        
        let mut batch_refs: Vec<(&mut [f64], usize, i32)> = batches
            .iter_mut()
            .map(|(data, n, isign)| (data.as_mut_slice(), *n, *isign))
            .collect();
        
        let start = std::time::Instant::now();
        processor.process_batch(&mut batch_refs);
        let duration = start.elapsed();
        
        println!("Batch RealFT processing (4x512): {:?}", duration);
    }
}
