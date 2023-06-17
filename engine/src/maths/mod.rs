pub mod activation;
mod high_freq_computation;
mod matrix;
pub mod matrix_ops;

pub use matrix::Matrix;

const LEAKY_RELU_VALUE: f64 = 1E-2;
const MULTITHREADED: bool = false;
