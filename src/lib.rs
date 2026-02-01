//! Parallel matrix multiplication library using Burn framework.
//!
//! Provides CPU and GPU workers that process matrix multiplications
//! concurrently using different backends.

mod dispatcher;
mod job;
mod workers;

pub use dispatcher::Dispatcher;
pub use job::{MatMulJob, MatMulResult};
