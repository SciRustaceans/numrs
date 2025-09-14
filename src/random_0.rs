// lib.rs
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// Constants from the original C code, converted to i64
const IA: i64 = 16807;
const IM: i64 = 2147483647;
const AM: f64 = 1.0 / IM as f64;
const IQ: i64 = 127773;
const IR: i64 = 2863;
const MASK: i64 = 123459876;

pub struct Ran0 {
    state: i64,
}

impl Ran0 {
    pub fn new(seed: i64) -> Self {
        let mut state = seed;
        // Apply mask to initialize state
        state ^= MASK;
        Self { state }
    }

    pub fn next(&mut self) -> f64 {
        let k = self.state / IQ;
        self.state = IA * (self.state - k * IQ) - IR * k;
        
        if self.state < 0 {
            self.state += IM;
        }
        
        let ans = AM * self.state as f64;
        // Reapply mask for next call
        self.state ^= MASK;
        
        ans
    }

    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next()
    }

    // Generate multiple values at once
    pub fn generate(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

// Thread-safe version using Mutex
pub struct ThreadSafeRan0 {
    inner: Mutex<Ran0>,
}

impl ThreadSafeRan0 {
    pub fn new(seed: i64) -> Self {
        Self {
            inner: Mutex::new(Ran0::new(seed)),
        }
    }

    pub fn next(&self) -> f64 {
        self.inner.lock().unwrap().next()
    }

    pub fn next_range(&self, min: f64, max: f64) -> f64 {
        self.inner.lock().unwrap().next_range(min, max)
    }
}

// Parallel random number generation
pub fn generate_parallel(seed: i64, n: usize, chunk_size: usize) -> Vec<f64> {
    // Create different seeds for each thread
    let seeds: Vec<i64> = (0..(n / chunk_size + 1))
        .map(|i| seed.wrapping_mul(i as i64 + 1))
        .collect();

    seeds.par_iter()
        .flat_map(|&s| {
            let mut rng = Ran0::new(s);
            rng.generate(chunk_size.min(n))
        })
        .take(n)
        .collect()
}

// Iterator implementation for seamless integration
impl Iterator for Ran0 {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        Some(Ran0::next(self))
    }
}

// Benchmarking and testing utilities
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ran0() {
        let mut rng = Ran0::new(12345);
        let values: Vec<f64> = rng.take(10).collect();
        assert_eq!(values.len(), 10);
        
        // All values should be in [0, 1)
        for &val in &values {
            assert!(val >= 0.0 && val < 1.0);
        }
        
        // Test thread-safe version
        let safe_rng = ThreadSafeRan0::new(12345);
        let safe_val = safe_rng.next();
        assert!(safe_val >= 0.0 && safe_val < 1.0);
    }

    #[test]
    fn test_parallel_generation() {
        let n = 1000;
        let values = generate_parallel(12345, n, 100);
        assert_eq!(values.len(), n);
        
        // All values should be in [0, 1)
        for &val in &values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }
}

// Additional utility functions
impl Ran0 {
    // Fill a slice with random values
    pub fn fill(&mut self, slice: &mut [f64]) {
        for item in slice.iter_mut() {
            *item = self.next();
        }
    }
    
    // Create a new RNG with a random seed from the system time
    #[cfg(feature = "std")]
    pub fn from_system_time() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        
        Self::new(seed)
    }
}
