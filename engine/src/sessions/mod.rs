mod dense_session;
use crate::networks::Network;
pub use dense_session::DenseSession;

pub trait Session<T: Network> {
    fn fit(&mut self) -> f64;
    fn train(&mut self);
    fn test(&mut self) -> f64;

    fn release_network(self) -> T;
}
