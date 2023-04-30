mod matrix;
pub mod activation;
pub mod matrix_ops;
mod high_freq_computation;

pub use matrix::Matrix;

const LEAKY_RELU_VALUE: f64 = 1E-2;
const MULTITHREADED: bool = false;