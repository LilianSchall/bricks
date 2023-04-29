use std::fs;
use std::str::FromStr;
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
    biases: Vec<Matrix>,
    raw_values: Vec<Matrix>,
    values: Vec<Matrix>,
    epsilon: f64,
}

impl DenseModel {
    pub fn new(activations: Vec<DenseActivation>, loss: Loss,
               shape: Vec<DenseShape>, epsilon: Option<f64>) -> DenseModel {
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
            mat = (&mat + &self.biases[i]).unwrap();
            self.raw_values[i + 1] = mat.clone();
            self.activations[i].apply(&mut mat, self.epsilon);
            self.values[i + 1] = mat;
        }
    }

    pub fn value(&self) -> Matrix {
        self.values[self.nb_layers - 1].clone()
    }

    pub fn online_back_propagate(&self, output: &Matrix) -> Vec<Matrix> {
        let mut deltas: Vec<Matrix> = Vec::with_capacity(self.nb_layers);
        for l in (1..self.nb_layers).rev() {
            let delta: Matrix;
            let mut d_z = self.raw_values[l].clone();
            self.activations[l - 1].derivate(&mut d_z, self.epsilon);
            if l == self.nb_layers - 1 {
                delta = self.loss.compute_differential_error(&self.values[l], output)
                    .hadamard_dot(&d_z).unwrap();
            } else {
                delta = (&self.weights[l].t() * &deltas[deltas.len() - 1]).unwrap().hadamard_dot(&d_z).unwrap();
            }
            deltas.push(delta);
        }
        deltas.reverse();
        deltas
    }

    pub fn update_weights(&mut self, deltas: Vec<Matrix>, learning_rate: f64) {
        for l in (1..self.nb_layers).rev() {
            self.biases[l - 1] = (&self.biases[l - 1] - &(&deltas[l - 1] * learning_rate)).unwrap();

            for j in 0..deltas[l - 1].len() {
                for k in 0..self.values[l - 1].len() {
                    let a = self.values[l - 1].get(k).unwrap();
                    let d = deltas[l - 1].get(j).unwrap();
                    let w = self.weights[l - 1].get_at(j, k).unwrap();
                    self.weights[l - 1].set_at(j, k, w - learning_rate * (a * d));
                }
            }
        }
    }

    pub fn load_model(path: String) -> DenseModel {
        let mut weights: Vec<Matrix> = vec![];
        let mut biases: Vec<Matrix> = vec![];
        let mut activations: Vec<DenseActivation> = vec![];
        let mut loss: Loss = Loss::CategoricalCrossEntropy;

        let mut shape: Vec<DenseShape> = vec![];

        let contents = fs::read_to_string(path).expect("Loading path is invalid");

        let lines = contents.split("\n").collect::<Vec<_>>();
        let mut phase: usize = 0;
        let mut shape_selector: usize = 0;
        for line in lines {
            match phase {
                0 => {
                    shape = line.split(" ")
                        .map(|value| DenseShape::one_d(value.parse::<usize>().unwrap()))
                        .collect::<Vec<_>>();
                    weights = Vec::with_capacity(shape.len() - 1);
                    biases = Vec::with_capacity(shape.len() - 1);
                }
                1 => activations = line.split(" ")
                    .map(|value| DenseActivation::from_str(value).unwrap())
                    .collect::<Vec<DenseActivation>>(),
                _ => {
                    if phase % 2 == 0 {
                        weights.push(Matrix::reshape(
                            line.split(" ")
                                .map(|value| value.parse::<f64>().unwrap())
                                .collect::<Vec<f64>>()
                            , shape[shape_selector].range
                            , shape[shape_selector + 1].range)
                            .unwrap());
                        shape_selector += 1;
                    } else {
                        biases.push(Matrix::reshape(
                            line.split(" ")
                                .map(|value| value.parse::<f64>().unwrap())
                                .collect::<Vec<f64>>()
                            , 1
                            , shape[shape_selector].range)
                            .unwrap());
                    }
                }
            }
            phase += 1;
        }
        let mut values = Vec::with_capacity(shape.len());
        let mut raw_values = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            values.push(Matrix::new(1, shape[i].range));
            raw_values.push(Matrix::new(1, shape[i].range));
        }

        DenseModel {
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

    pub fn save_model(&self, path: String) {
        let mut content: String = "".to_owned();

        for i in 0..self.values.len() {
            if i != 0 {
                content.push_str(" ");
            }
            content.push_str(self.values[i].h.to_string().as_str());
        }
        content.push_str("\n");
        for i in 0..self.activations.len() {
            if i != 0 {
                content.push_str(" ");
            }
            content.push_str(self.activations[i].to_string().as_str());
        }
        content.push_str("\n");

        concat_weights_and_bias(&mut content, &self.weights, &self.biases);

        fs::write(path, content).expect("Could not save the model at the given path.");
    }

}

fn concat_weights_and_bias(c: &mut String, weights: &Vec<Matrix>, biases: &Vec<Matrix>) {
    for i in 0..weights.len() {
        if i != 0 {
            c.push_str("\n");
        }
        c.push_str(&weights[i].to_string());
        c.push_str("\n");
        c.push_str(&biases[i].to_string());
    }
}