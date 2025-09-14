// lib.rs
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

// Constants from the original C code, converted to i64
const IM1: i64 = 2147483563;
const IM2: i64 = 2147483399;
const AM: f64 = 1.0 / IM1 as f64;
const IMM1: i64 = IM1 - 1;
const IA1: i64 = 40014;
const IA2: i64 = 40692;
const IQ1: i64 = 53668;
const IQ2: i64 = 52774;
const IR1: i64 = 12211;
const IR2: i64 = 3791;
const NTAB: usize = 32;
const NDIV: i64 = 1 + IMM1 / NTAB as i64;
const EPS: f64 = 1.2e-7;
const RNMX: f64 = 1.0 - EPS;

pub struct Ran2 {
    idum1: i64,
    idum2: i64,
    iy: i64,
    iv: [i64; NTAB],
    initialized: bool,
}

impl Ran2 {
    pub fn new(seed: i64) -> Self {
        let mut idum1 = if seed <= 0 {
            if -seed < 1 { 1 } else { -seed }
        } else {
            seed
        };

        let mut idum2 = idum1;
        let mut iv = [0; NTAB];
        let mut iy = 0;

        // Initialize the shuffle table
        for j in (0..NTAB + 7).rev() {
            let k = idum1 / IQ1;
            idum1 = IA1 * (idum1 - k * IQ1) - k * IR1;
            if idum1 < 0 {
                idum1 += IM1;
            }
            if j < NTAB {
                iv[j] = idum1;
            }
        }
        iy = iv[0];

        Self {
            idum1,
            idum2,
            iy,
            iv,
            initialized: true,
        }
    }

    pub fn next(&mut self) -> f64 {
        // First generator
        let k1 = self.idum1 / IQ1;
        self.idum1 = IA1 * (self.idum1 - k1 * IQ1) - k1 * IR1;
        if self.idum1 < 0 {
            self.idum1 += IM1;
        }

        // Second generator
        let k2 = self.idum2 / IQ2;
        self.idum2 = IA2 * (self.idum2 - k2 * IQ2) - k2 * IR2;
        if self.idum2 < 0 {
            self.idum2 += IM2;
        }

        // Combine generators using shuffle table
        let j = (self.iy / NDIV) as usize;
        self.iy = self.iv[j].wrapping_sub(self.idum2);
        self.iv[j] = self.idum1;

        if self.iy < 1 {
            self.iy += IMM1;
        }

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

    // Get the current internal state (for debugging/serialization)
    pub fn get_state(&self) -> (i64, i64, i64, [i64; NTAB]) {
        (self.idum1, self.idum2, self.iy, self.iv)
    }

    // Restore from a saved state
    pub fn from_state(idum1: i64, idum2: i64, iy: i64, iv: [i64; NTAB]) -> Self {
        Self {
            idum1,
            idum2,
            iy,
            iv,
            initialized: true,
        }
    }
}

// Thread-safe version using Mutex
pub struct ThreadSafeRan2 {
    inner: Mutex<Ran2>,
}

impl ThreadSafeRan2 {
    pub fn new(seed: i64) -> Self {
        Self {
            inner: Mutex::new(Ran2::new(seed)),
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

    pub fn get_state(&self) -> (i64, i64, i64, [i64; NTAB]) {
        self.inner.lock().unwrap().get_state()
    }
}

// Global thread-local instance for convenience
thread_local! {
    static THREAD_RNG: std::cell::RefCell<Ran2> = std::cell::RefCell::new(Ran2::new(123456789));
}

// Convenience functions for global RNG access
pub fn random() -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next())
}

pub fn random_range(min: f64, max: f64) -> f64 {
    THREAD_RNG.with(|rng| rng.borrow_mut().next_range(min, max))
}

pub fn set_seed(seed: i64) {
    THREAD_RNG.with(|rng| *rng.borrow_mut() = Ran2::new(seed));
}

// Iterator implementation
impl Iterator for Ran2 {
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
    fn test_ran2_initialization() {
        let mut rng = Ran2::new(-12345);
        let values: Vec<f64> = rng.take(10).collect();
        assert_eq!(values.len(), 10);
        
        // All values should be in [0, 1)
        for &val in &values {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_ran2_range() {
        let mut rng = Ran2::new(12345);
        let value = rng.next_range(5.0, 10.0);
        assert!(value >= 5.0 && value < 10.0);
    }

    #[test]
    fn test_ran2_fill() {
        let mut rng = Ran2::new(12345);
        let mut arr = [0.0; 100];
        rng.fill(&mut arr);
        
        for &val in &arr {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_thread_safe_ran2() {
        let safe_rng = ThreadSafeRan2::new(12345);
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
    fn test_ran2_sequence() {
        let mut rng1 = Ran2::new(42);
        let mut rng2 = Ran2::new(42);
        
        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_ran2_state_serialization() {
        let mut rng1 = Ran2::new(123);
        let _ = rng1.next(); // Advance state
        
        let state = rng1.get_state();
        let mut rng2 = Ran2::from_state(state.0, state.1, state.2, state.3);
        
        // Should continue from the same state
        for _ in 0..10 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_ran2_statistical_properties() {
        let mut rng = Ran2::new(999);
        let values = rng.generate(10000);
        
        // Basic statistical tests
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((mean - 0.5).abs() < 0.01, "Mean should be around 0.5");
        
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!((variance - 1.0/12.0).abs() < 0.01, "Variance should be around 1/12");
    }
}

// Additional utility functions
impl Ran2 {
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

    // Generate random integers in a range
    pub fn next_int(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min + 1) as f64;
        min + (self.next() * range) as i64
    }

    // Generate random booleans
    pub fn next_bool(&mut self) -> bool {
        self.next() < 0.5
    }
}

// Parallel generation using different seeds
pub fn generate_parallel_ran2(seed: i64, n: usize, chunk_size: usize) -> Vec<f64> {
    let seeds: Vec<i64> = (0..(n / chunk_size + 1))
        .map(|i| seed.wrapping_mul(i as i64 + 1))
        .collect();

    seeds.into_iter()
        .flat_map(|s| {
            let mut rng = Ran2::new(s);
            rng.generate(chunk_size.min(n))
        })
        .take(n)
        .collect()
}

// Specialized versions for common distributions
impl Ran2 {
    // Generate normally distributed random numbers using Box-Muller transform
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next();
        let u2 = self.next();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z0
    }

    // Generate exponentially distributed random numbers
    pub fn next_exponential(&mut self, lambda: f64) -> f64 {
        -self.next().ln() / lambda
    }
}
