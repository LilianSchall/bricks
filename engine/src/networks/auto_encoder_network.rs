use std::fs;
use crate::activations::DenseActivation;
use crate::losses::Loss;
use crate::maths::Matrix;
use crate::networks::{DEFAULT_EPSILON_VALUE, Network};
use crate::networks::network_operations::{feed_forward_generics, load_network_generics};
use crate::shapes::DenseShape;

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
    pub fn new(activations: Vec<DenseActivation>, loss: Loss,
               shape: Vec<DenseShape>, epsilon: Option<f64>) -> Self {

        assert_eq!(activations.len(), 3);
        assert_eq!(shape.len(), 3);

        let mut weights = Vec::with_capacity(shape.len() - 1);
        let mut biases = Vec::with_capacity(shape.len() - 1);
        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        for i in 0..(shape.len() - 1) {
            weights.push(Matrix::random(shape[i].range, shape[i + 1].range));
            biases.push(Matrix::random(1, shape[i + 1].range));
        }

        AutoEncoderNetwork {
            loss,
            activations,
            weights,
            biases,
            raw_values,
            values,
            epsilon: epsilon.unwrap_or(DEFAULT_EPSILON_VALUE),
        }
    }


    // todo: refactoring of backpropagation for dense_network and auto_encoder_network
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
        let mut weights: Vec<Matrix> = Vec::with_capacity(1);
        let mut biases: Vec<Matrix> = Vec::with_capacity(1);
        let mut activations: Vec<DenseActivation> = Vec::with_capacity(2);
        let mut loss: Loss = Loss::CategoricalCrossEntropy;
        let mut shape: Vec<DenseShape> = Vec::with_capacity(2);

        let contents = fs::read_to_string(path).expect("Loading path is invalid");
        let lines = contents.split("\n").collect::<Vec<_>>();

        load_network_generics(&mut weights, &mut biases, &mut activations, &mut loss, &mut shape, lines);

        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        AutoEncoderNetwork {
            loss,
            activations,
            weights,
            biases,
            raw_values,
            values,
            epsilon: DEFAULT_EPSILON_VALUE,
        }
    }

    fn save_network(&self, path: &str) {
        todo!()
    }
}