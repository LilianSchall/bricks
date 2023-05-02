use crate::activations::DenseActivation;
use crate::losses::Loss;
use crate::maths::Matrix;
use crate::networks::Network;
use crate::networks::network_operations::feed_forward_generics;

pub struct AutoEncoderNetwork {
    // There is no number of layers, since an auto-encoder is considered of three layers
    // ,one of those is disposable after training: so there is two weights matrices:
    // the encoder weights that link the input layer to the hidden layer,
    // and the decoder weights that link the hidden layer to the output layer.
    // As a dense network, those layers still have each one an activation function
    // as well as a loss function for backpropagation

    pub loss: Loss,
    activations: Vec<DenseActivation>,
    weights: Vec<Matrix>,

    biases: Vec<Matrix>,
    raw_values: Vec<Matrix>,
    values: Vec<Matrix>,
    epsilon: f64,
}

impl AutoEncoderNetwork {

}

impl Network for AutoEncoderNetwork {
    fn feed_forward(&mut self, input: &Matrix) {
        if input.h != self.values[0].h {
            return;
        }

        self.values[0] = input.clone();
        self.raw_values[0] = input.clone();
        let nb_layers = self.values.len();

        feed_forward_generics(&mut self.values, &mut self.raw_values, &self.activations, &self.weights,
                              &self.biases, nb_layers, self.epsilon);
    }

    fn value(&self) -> Matrix {
        self.values[self.values.len() - 1].clone()
    }

    fn load_network(path: &str) -> Self {
        todo!()
    }

    fn save_network(&self, path: &str) {
        todo!()
    }
}