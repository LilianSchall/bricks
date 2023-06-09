use crate::activations::DenseActivation;
use crate::losses::Loss;
use crate::maths::Matrix;
use crate::networks::network_operations::{
    feed_forward_generics, load_network_generics, back_propagation_generics,
    save_network_generics, update_weights_generics,
};
use crate::networks::{Network, SupervisedNetwork, DEFAULT_EPSILON_VALUE};
use crate::shapes::DenseShape;
use std::fs;

pub struct AutoEncoderNetwork {
    nb_layers: usize,
    pub loss: Loss,
    activations: Vec<DenseActivation>,
    weights: Vec<Matrix>,

    biases: Vec<Matrix>,
    raw_values: Vec<Matrix>,
    values: Vec<Matrix>,
    epsilon: f64,
}

impl AutoEncoderNetwork {
    pub fn new(
        activations: Vec<DenseActivation>,
        loss: Loss,
        shape: Vec<DenseShape>,
        epsilon: Option<f64>,
    ) -> Self {
        assert_eq!(activations.len(), shape.len() - 1);

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
            nb_layers: shape.len(),
            loss,
            activations,
            weights,
            biases,
            raw_values,
            values,
            epsilon: epsilon.unwrap_or(DEFAULT_EPSILON_VALUE),
        }
    }

    pub fn online_back_propagate(&self, output: &Matrix) -> Vec<Matrix> {
        let mut deltas: Vec<Matrix> = Vec::with_capacity(self.nb_layers);

        back_propagation_generics(
            &mut deltas,
            &self.activations,
            &self.values,
            &self.raw_values,
            &self.loss,
            &self.weights,
            self.nb_layers,
            self.epsilon,
            output,
        );

        deltas
    }
}

impl SupervisedNetwork for AutoEncoderNetwork {
    fn feed_backward(&self, output: &Matrix) -> Vec<Matrix> {
        self.online_back_propagate(output)
    }

    fn update_weights(&mut self, deltas: Vec<Matrix>, learning_rate: f64) {
        update_weights_generics(
            deltas,
            learning_rate,
            &self.values,
            &mut self.weights,
            &mut self.biases,
            self.nb_layers,
        );
    }
}

impl Network for AutoEncoderNetwork {
    fn feed_forward(&mut self, input: &Matrix) {
        if input.h != self.values[0].h {
            return;
        }

        self.values[0] = input.clone();
        self.raw_values[0] = input.clone();
        let nb_layers = self.values.len();

        feed_forward_generics(
            &mut self.values,
            &mut self.raw_values,
            &self.activations,
            &self.weights,
            &self.biases,
            nb_layers,
            self.epsilon,
        );
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

        load_network_generics(
            &mut weights,
            &mut biases,
            &mut activations,
            &mut loss,
            &mut shape,
            lines,
        );

        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        AutoEncoderNetwork {
            nb_layers: shape.len(),
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
        let shape = self.values.iter().map(|v| DenseShape::one_d(v.h)).collect();
        save_network_generics(
            path,
            shape,
            &self.activations,
            &self.loss,
            &self.weights,
            &self.biases,
        );
    }
}
