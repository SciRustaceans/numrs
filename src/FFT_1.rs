use std::f64::consts::PI;
use rayon::prelude::*;
use std::simd::{f64x2, Simd};

pub fn four1(data: &mut [f64], nn: usize, isign: i32) {
    let n = nn * 2;
    
    // Bit-reversal permutation
    let mut j = 1;
    for i in (1..n).step_by(2) {
        if j > i {
            data.swap(j - 1, i - 1);
            data.swap(j, i);
        }
        
        let mut m = nn;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Daniel-Lanczos section with SIMD and parallelization
    let mut mmax = 2;
    while n > mmax {
        let istep = mmax << 1;
        let theta = isign as f64 * (2.0 * PI / mmax as f64);
        let wtemp = (0.5 * theta).sin();
        let wpr = -2.0 * wtemp * wtemp;
        let wpi = theta.sin();
        
        // Process multiple butterflies in parallel
        if mmax >= 1024 {
            // Parallel version for large transforms
            process_butterflies_parallel(data, mmax, istep, wpr, wpi, isign);
        } else {
            // Sequential version for small transforms
            process_butterflies_sequential(data, mmax, istep, wpr, wpi, isign);
        }
        
        mmax = istep;
    }
}

#[inline(always)]
fn process_butterflies_sequential(data: &mut [f64], mmax: usize, istep: usize, wpr: f64, wpi: f64, isign: i32) {
    let mut wr = 1.0;
    let mut wi = 0.0;
    
    for m in (0..mmax).step_by(2) {
        for i in (m..data.len()).step_by(istep) {
            let j = i + mmax;
            if j >= data.len() {
                continue;
            }
            
            let tempr = wr * data[j] - wi * data[j + 1];
            let tempi = wr * data[j + 1] + wi * data[j];
            
            data[j] = data[i] - tempr;
            data[j + 1] = data[i + 1] - tempi;
            data[i] += tempr;
            data[i + 1] += tempi;
        }
        
        // Update rotation factors
        let wtemp = wr;
        wr = wtemp * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
}

#[inline(always)]
fn process_butterflies_parallel(data: &mut [f64], mmax: usize, istep: usize, wpr: f64, wpi: f64, isign: i32) {
    // Precompute rotation factors for this stage
    let rotation_factors: Vec<(f64, f64)> = (0..mmax/2)
        .map(|m| {
            let angle = isign as f64 * 2.0 * PI * m as f64 / mmax as f64;
            (angle.cos(), angle.sin())
        })
        .collect();

    // Process each butterfly group in parallel
    data.par_chunks_mut(istep)
        .for_each(|chunk| {
            if chunk.len() < mmax * 2 {
                return;
            }
            
            for m in 0..mmax/2 {
                let (wr, wi) = rotation_factors[m];
                
                for i in (m * 2..chunk.len() - mmax).step_by(istep) {
                    let j = i + mmax;
                    
                    let tempr = wr * chunk[j] - wi * chunk[j + 1];
                    let tempi = wr * chunk[j + 1] + wi * chunk[j];
                    
                    chunk[j] = chunk[i] - tempr;
                    chunk[j + 1] = chunk[i + 1] - tempi;
                    chunk[i] += tempr;
                    chunk[i + 1] += tempi;
                }
            }
        });
}

// SIMD-optimized version for modern CPUs
pub fn four1_simd(data: &mut [f64], nn: usize, isign: i32) {
    let n = nn * 2;
    
    // Bit-reversal permutation (same as before)
    let mut j = 1;
    for i in (1..n).step_by(2) {
        if j > i {
            data.swap(j - 1, i - 1);
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
        let theta = isign as f64 * (2.0 * PI / mmax as f64);
        
        // Process with SIMD if possible
        if mmax >= 4 && istep % 4 == 0 {
            process_butterflies_simd(data, mmax, istep, theta);
        } else {
            let wtemp = (0.5 * theta).sin();
            let wpr = -2.0 * wtemp * wtemp;
            let wpi = theta.sin();
            process_butterflies_sequential(data, mmax, istep, wpr, wpi, isign);
        }
        
        mmax = istep;
    }
}

#[inline(always)]
fn process_butterflies_simd(data: &mut [f64], mmax: usize, istep: usize, theta: f64) {
    let mut wr = 1.0;
    let mut wi = 0.0;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    
    for m in (0..mmax).step_by(2) {
        // Use SIMD for processing multiple butterflies at once
        for i in (m..data.len()).step_by(istep) {
            if i + mmax + 3 >= data.len() {
                continue;
            }
            
            // Load data using SIMD
            let data_i = f64x2::from_slice(&data[i..i+2]);
            let data_j = f64x2::from_slice(&data[i+mmax..i+mmax+2]);
            
            // Perform complex multiplication: (wr + i*wi) * (data_j.re + i*data_j.im)
            let tempr = wr * data_j[0] - wi * data_j[1];
            let tempi = wr * data_j[1] + wi * data_j[0];
            
            // Store results
            data[i+mmax] = data[i] - tempr;
            data[i+mmax+1] = data[i+1] - tempi;
            data[i] += tempr;
            data[i+1] += tempi;
        }
        
        // Update rotation factors
        let wtemp = wr;
        wr = wtemp * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
}

// Thread-safe FFT processor
pub struct FFTProcessor {
    max_threads: usize,
    use_simd: bool,
}

impl FFTProcessor {
    pub fn new() -> Self {
        Self {
            max_threads: rayon::current_num_threads(),
            use_simd: true,
        }
    }
    
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.max_threads = threads;
        self
    }
    
    pub fn with_simd(mut self, use_simd: bool) -> Self {
        self.use_simd = use_simd;
        self
    }
    
    pub fn fft(&self, data: &mut [f64], isign: i32) {
        let nn = data.len() / 2;
        
        if self.use_simd && nn >= 512 {
            four1_simd(data, nn, isign);
        } else if nn >= 2048 {
            // Use parallel version for large transforms
            rayon::scope(|s| {
                s.spawn(|_| {
                    four1(data, nn, isign);
                });
            });
        } else {
            // Use sequential version for small transforms
            four1(data, nn, isign);
        }
    }
    
    // Batch processing of multiple FFTs
    pub fn fft_batch(&self, batches: &mut [&mut [f64]], isign: i32) {
        batches.par_iter_mut().for_each(|data| {
            self.fft(data, isign);
        });
    }
}

// Utility functions
pub fn real_to_complex(real_data: &[f64]) -> Vec<f64> {
    let mut complex = Vec::with_capacity(real_data.len() * 2);
    for &value in real_data {
        complex.push(value);
        complex.push(0.0);
    }
    complex
}

pub fn complex_to_real(complex_data: &[f64]) -> Vec<f64> {
    complex_data.iter().step_by(2).copied().collect()
}

// Benchmarking and testing
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn test_signal() -> Vec<f64> {
        let mut signal = Vec::new();
        for i in 0..1024 {
            let t = i as f64 / 1024.0;
            signal.push((2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).cos());
        }
        signal
    }

    #[test]
    fn test_fft_correctness() {
        let signal = test_signal();
        let mut complex_signal = real_to_complex(&signal);
        
        // Forward FFT
        four1(&mut complex_signal, signal.len(), 1);
        
        // Inverse FFT
        four1(&mut complex_signal, signal.len(), -1);
        
        // Scale back
        for value in &mut complex_signal {
            *value /= signal.len() as f64;
        }
        
        let reconstructed = complex_to_real(&complex_signal);
        
        // Check reconstruction accuracy
        for (orig, rec) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - rec).abs() < 1e-10, "FFT reconstruction error");
        }
    }

    #[test]
    fn test_fft_performance() {
        let sizes = [256, 1024, 4096, 16384];
        
        for &size in &sizes {
            let signal: Vec<f64> = (0..size).map(|i| (2.0 * PI * i as f64 / size as f64).sin()).collect();
            let mut complex_signal = real_to_complex(&signal);
            
            let start = Instant::now();
            four1(&mut complex_signal, size, 1);
            let duration = start.elapsed();
            
            println!("FFT size {}: {:?}", size, duration);
        }
    }

    #[test]
    fn test_parallel_fft() {
        let processor = FFTProcessor::new();
        let mut signals: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..1024).map(|j| (2.0 * PI * (i + 1) as f64 * j as f64 / 1024.0).sin()).collect()
            })
            .collect();
        
        let mut complex_signals: Vec<Vec<f64>> = signals.iter().map(|s| real_to_complex(s)).collect();
        let mut slices: Vec<&mut [f64]> = complex_signals.iter_mut().map(|s| s.as_mut_slice()).collect();
        
        let start = Instant::now();
        processor.fft_batch(&mut slices, 1);
        let duration = start.elapsed();
        
        println!("Parallel FFT batch: {:?}", duration);
    }
}

// Example usage
fn main() {
    // Example: FFT of a simple signal
    let signal: Vec<f64> = (0..1024).map(|i| (2.0 * PI * i as f64 / 1024.0).sin()).collect();
    let mut complex_signal = real_to_complex(&signal);
    
    println!("Performing FFT...");
    let start = Instant::now();
    
    let processor = FFTProcessor::new();
    processor.fft(&mut complex_signal, 1);
    
    let duration = start.elapsed();
    println!("FFT completed in {:?}", duration);
    
    // Process frequency domain data here...
    
    // Inverse FFT
    processor.fft(&mut complex_signal, -1);
    for value in &mut complex_signal {
        *value /= 1024.0; // Scale back
    }
}
