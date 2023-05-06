mod dense_network;
mod auto_encoder_network;
mod network_operations;

pub use dense_network::DenseNetwork;
use crate::maths::Matrix;

pub const DEFAULT_EPSILON_VALUE: f64 = 1E-6;

pub trait Network {
    fn feed_forward(&mut self, input: &Matrix);

    fn value(&self) -> Matrix;

    fn load_network(path: &str) -> Self;
    fn save_network(&self, path: &str);
}

pub trait SupervisedNetwork {
    fn feed_backward(&self, output: &Matrix) -> Vec<Matrix>;
    fn update_weights(&mut self, deltas: Vec<Matrix>, learning_rate: f64);
}