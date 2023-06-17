mod dense_network;
mod network_operations;

use crate::maths::Matrix;
pub use dense_network::DenseNetwork;

pub const DEFAULT_EPSILON_VALUE: f64 = 1E-6;

pub trait Network {
    fn feed_forward(&mut self, input: &Matrix);

    fn value(&self) -> Matrix;
    fn output_shape(&self) -> (usize, usize);

    fn load_network(path: &str) -> Self;
    fn save_network(&self, path: &str);
}

pub trait SupervisedNetwork {
    fn compute_output_delta(&self, output: &Matrix) -> Matrix;
    fn feed_backward(&self, output_delta: Matrix) -> Vec<Matrix>;
    fn update_weights(&mut self, deltas: Vec<Matrix>, learning_rate: f64);
}
