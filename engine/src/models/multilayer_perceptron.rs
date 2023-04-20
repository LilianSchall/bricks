use crate::activations::multilayer_activation::MultilayerActivation;
use crate::maths::matrix::Matrix;
use crate::losses::losses::Loss;
use crate::shapes::multilayer_shape::MultilayerShape;

pub struct MultilayerPerceptron {
    nb_layers: usize,
    pub loss: Loss,
    activations: Vec<MultilayerActivation>,
    weights: Vec<Matrix>,

    // line matrix, represents the bias of the hidden layers and output layer
    biases: Matrix,
    raw_values: Vec<Matrix>,
    values: Vec<Matrix>,
}

impl MultilayerPerceptron {
    pub fn new(activations: Vec<MultilayerActivation>, loss: Loss,
               shape: Vec<MultilayerShape>) -> MultilayerPerceptron {

        let mut weights = Vec::with_capacity(shape.len() - 1);
        let mut biases = Matrix::new(shape.len() - 1, 1);
        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        for i in 0..(shape.len() - 1) {
            weights.push(Matrix::random(shape[i].range, shape[i + 1].range));
        }

        MultilayerPerceptron {
            nb_layers: shape.len(),
            loss,
            activations,
            weights,
            biases,
            raw_values,
            values
        }
    }
}