use crate::activations::dense_activation::DenseActivation;
use crate::maths::matrix::Matrix;
use crate::losses::losses::Loss;
use crate::models::DEFAULT_EPSILON_VALUE;
use crate::shapes::dense_shape::DenseShape;

pub struct DenseModel {
    nb_layers: usize,
    pub loss: Loss,
    activations: Vec<DenseActivation>,
    weights: Vec<Matrix>,

    // line matrix, represents the bias of the hidden layers and output layer
    biases: Matrix,
    raw_values: Vec<Matrix>,
    values: Vec<Matrix>,
    epsilon: f64,
}

impl DenseModel {
    pub fn new(activations: Vec<DenseActivation>, loss: Loss,
               shape: Vec<DenseShape>, epsilon: Option<f64>) -> DenseModel {
        let mut weights = Vec::with_capacity(shape.len() - 1);
        let biases = Matrix::random(shape.len() - 1, 1);
        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        for i in 0..(shape.len() - 1) {
            weights.push(Matrix::random(shape[i].range, shape[i + 1].range));
        }

        DenseModel {
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

    pub fn feed_forward(&mut self, input: &Matrix) {
        if input.h != self.values[0].h {
            return;
        }

        self.values[0] = input.clone();
        self.raw_values[0] = input.clone();

        for i in 0..(self.nb_layers - 1) {
            let mut mat = (&self.weights[i] * &self.values[i]).unwrap();
            mat = mat + self.biases.get(i).unwrap();
            self.raw_values[i + 1] = mat.clone();
            self.activations[i].apply(&mut mat, self.epsilon);
            self.values[i + 1] = mat;
        }
    }

    pub fn back_propagate(&mut self, output: &Matrix) -> Vec<Matrix> {
        let mut deltas: Vec<Matrix> = Vec::with_capacity(self.nb_layers);

        for l in (1..self.nb_layers).rev() {

        }

        deltas
    }
}