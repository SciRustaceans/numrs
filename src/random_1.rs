// lib.rs
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

// Constants from the original C code, converted to i64
const IA: i64 = 16807;
const IM: i64 = 2147483647;
const AM: f64 = 1.0 / IM as f64;
const IQ: i64 = 127773;
const IR: i64 = 2836;
const NTAB: usize = 32;
const NDIV: i64 = 1 + (IM - 1) / NTAB as i64;
const EPS: f64 = 1.2e-7;
const RNMX: f64 = 1.0 - EPS;

pub struct Ran1 {
    state: i64,
    iy: i64,
    iv: [i64; NTAB],
    initialized: bool,
}

impl Ran1 {
    pub fn new(seed: i64) -> Self {
        let mut state = if seed <= 0 {
            if -seed < 1 { 1 } else { -seed }
        } else {
            seed
        };

        let mut iv = [0; NTAB];
        let mut iy = 0;

        // Initialize the shuffle table
        for j in (0..NTAB + 7).rev() {
            let k = state / IQ;
            state = IA * (state - k * IQ) - IR * k;
            if state < 0 {
                state += IM;
            }
            if j < NTAB {
                iv[j] = state;
            }
        }
        iy = iv[0];

        Self {
            state,
            iy,
            iv,
            initialized: true,
        }
    }

    pub fn next(&mut self) -> f64 {
        let k = self.state / IQ;
        self.state = IA * (self.state - k * IQ) - IR * k;
        if self.state < 0 {
            self.state += IM;
        }

        let j = (self.iy / NDIV) as usize;
        self.iy = self.iv[j];
        self.iv[j] = self.state;

        let temp = AM * self.iy as f64;
        if temp > RNMX {
            RNMX
        } else {
            temp
        }
    }

    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next()
    }

    // Generate multiple values at once
    pub fn generate(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }

    // Fill a slice with random values
    pub fn fill(&mut self, slice: &mut [f64]) {
        for item in slice.iter_mut() {
            *item = self.next();
        }
    }
}

// Thread-safe version using Mutex
pub struct ThreadSafeRan1 {
    inner: Mutex<Ran1>,
}

impl ThreadSafeRan1 {
    pub fn new(seed: i64) -> Self {
        Self {
            inner: Mutex::new(Ran1::new(seed)),
        }
    }

    pub fn next(&self) -> f64 {
        self.inner.lock().unwrap().next()
    }

    pub fn next_range(&self, min: f64, max: f64) -> f64 {
        self.inner.lock().unwrap().next_range(min, max)
    }

    pub fn generate(&self, n: usize) -> Vec<f64> {
        self.inner.lock().unwrap().generate(n)
    }
}

// Global thread-local instance for convenience
thread_local! {
    static THREAD_RNG: std::cell::RefCell<Ran1> = std::cell::RefCell::new(Ran1::new(12345));
}

// Convenience functions for global RNG access
pub fn random() -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next())
}

pub fn random_range(min: f64, max: f64) -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next_range(min, max))
}

pub fn set_seed(seed: i64) {
    THREAD_RNG.with(|rng| *rng.borrow_mut() = Ran1::new(seed));
}

// Iterator implementation
impl Iterator for Ran1 {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        Some(self.next())
    }
}

// Benchmarking and testing utilities
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ran1_initialization() {
        let mut rng = Ran1::new(-12345);
        let values: Vec<f64> = rng.take(10).collect();
        assert_eq!(values.len(), 10);
        
        // All values should be in [0, 1)
        for &val in &values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_ran1_range() {
        let mut rng = Ran1::new(12345);
        let value = rng.next_range(5.0, 10.0);
        assert!(value >= 5.0 && value < 10.0);
    }

    #[test]
    fn test_ran1_fill() {
        let mut rng = Ran1::new(12345);
        let mut arr = [0.0; 100];
        rng.fill(&mut arr);
        
        for &val in &arr {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_thread_safe_ran1() {
        let safe_rng = ThreadSafeRan1::new(12345);
        let value = safe_rng.next();
        assert!(value >= 0.0 && value < 1.0);
        
        let values = safe_rng.generate(5);
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn test_global_rng() {
        set_seed(54321);
        let value = random();
        assert!(value >= 0.0 && value < 1.0);
        
        let ranged = random_range(1.0, 2.0);
        assert!(ranged >= 1.0 && ranged < 2.0);
    }

    #[test]
    fn test_ran1_sequence() {
        let mut rng1 = Ran1::new(42);
        let mut rng2 = Ran1::new(42);
        
        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }
}

// Additional utility functions
impl Ran1 {
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

    // Get the current internal state (for debugging/serialization)
    pub fn get_state(&self) -> (i64, i64, [i64; NTAB]) {
        (self.state, self.iy, self.iv)
    }

    // Restore from a saved state
    pub fn from_state(state: i64, iy: i64, iv: [i64; NTAB]) -> Self {
        Self {
            state,
            iy,
            iv,
            initialized: true,
        }
    }
}

// Parallel generation using different seeds
pub fn generate_parallel_ran1(seed: i64, n: usize, chunk_size: usize) -> Vec<f64> {
    let seeds: Vec<i64> = (0..(n / chunk_size + 1))
        .map(|i| seed.wrapping_mul(i as i64 + 1))
        .collect();

    seeds.into_iter()
        .flat_map(|s| {
            let mut rng = Ran1::new(s);
            rng.generate(chunk_size.min(n))
        })
        .take(n)
        .collect()
}
