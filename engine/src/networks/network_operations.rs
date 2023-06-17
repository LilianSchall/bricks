use crate::activations::DenseActivation;
use crate::losses::Loss;
use crate::maths::Matrix;
use crate::shapes::DenseShape;
use std::fs;
use std::str::FromStr;

pub fn feed_forward_generics(
    values: &mut Vec<Matrix>,
    raw_values: &mut Vec<Matrix>,
    activations: &Vec<DenseActivation>,
    weights: &Vec<Matrix>,
    biases: &Vec<Matrix>,
    nb_layers: usize,
) {
    for i in 0..(nb_layers - 1) {
        let mut mat = &weights[i] * &values[i];
        mat = &mat + &biases[i];
        raw_values[i + 1] = mat.clone();
        activations[i].apply(&mut mat);
        values[i + 1] = mat;
    }
}

pub fn load_network_generics(
    weights: &mut Vec<Matrix>,
    biases: &mut Vec<Matrix>,
    activations: &mut Vec<DenseActivation>,
    loss: &mut Loss,
    shape: &mut Vec<DenseShape>,
    lines: Vec<&str>,
) {
    let mut phase: usize = 0;
    let mut shape_selector: usize = 0;
    let nb_lines = lines.len();
    for line in lines {
        if phase == nb_lines - 1 {
            *loss = Loss::from_str(line).unwrap();
            break;
        }

        match phase {
            0 => {
                *shape = line
                    .split(" ")
                    .map(|value| DenseShape::one_d(value.parse::<usize>().unwrap()))
                    .collect::<Vec<_>>();
                *weights = Vec::with_capacity(shape.len() - 1);
                *biases = Vec::with_capacity(shape.len() - 1);
            }
            1 => {
                *activations = line
                    .split(" ")
                    .map(|value| DenseActivation::from_str(value).unwrap())
                    .collect::<Vec<DenseActivation>>()
            }
            _ => {
                if phase % 2 == 0 {
                    weights.push(Matrix::reshape(
                        line.split(" ")
                            .map(|value| value.parse::<f64>().unwrap())
                            .collect::<Vec<f64>>(),
                        shape[shape_selector].range,
                        shape[shape_selector + 1].range,
                    ));
                    shape_selector += 1;
                } else {
                    biases.push(Matrix::reshape(
                        line.split(" ")
                            .map(|value| value.parse::<f64>().unwrap())
                            .collect::<Vec<f64>>(),
                        1,
                        shape[shape_selector].range,
                    ));
                }
            }
        }
        phase += 1;
    }
}

pub fn save_network_generics(
    path: &str,
    shape: Vec<DenseShape>,
    activations: &Vec<DenseActivation>,
    loss: &Loss,
    weights: &Vec<Matrix>,
    biases: &Vec<Matrix>,
) {
    let mut content: String = "".to_owned();

    for i in 0..shape.len() {
        if i != 0 {
            content.push_str(" ");
        }
        content.push_str(shape[i].range.to_string().as_str());
    }
    content.push_str("\n");
    for i in 0..activations.len() {
        if i != 0 {
            content.push_str(" ");
        }
        content.push_str(activations[i].to_string().as_str());
    }
    content.push_str("\n");

    concat_weights_and_bias(&mut content, &weights, &biases);
    content.push_str("\n");
    content.push_str(&loss.to_string());

    fs::write(path, content).expect("Could not save the network at the given path.");
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

pub fn compute_output_delta_generics(
    activation: &DenseActivation,
    value: &Matrix,
    loss: &Loss,
    output: &Matrix,
) -> Matrix {
    let mut d_z = value.clone();
    activation.derivative(&mut d_z);
    let d= loss.compute_differential_error(value, output).hadamard_dot(&d_z);
    d
}

pub fn back_propagation_generics(
    deltas: &mut Vec<Matrix>,
    activations: &Vec<DenseActivation>,
    raw_values: &Vec<Matrix>,
    weights: &Vec<Matrix>,
    nb_layers: usize,
) {
    for l in (1..nb_layers - 1).rev() {
        let delta: Matrix;
        let mut d_z = raw_values[l].clone();
        activations[l - 1].derivative(&mut d_z);
        delta = (&weights[l].t() * &deltas[deltas.len() - 1]).hadamard_dot(&d_z);
        deltas.push(delta);
    }
    deltas.reverse();
}

pub fn update_weights_generics(
    deltas: Vec<Matrix>,
    learning_rate: f64,
    values: &Vec<Matrix>,
    weights: &mut Vec<Matrix>,
    biases: &mut Vec<Matrix>,
    nb_layers: usize,
    epsilon: f64,
) {
    for l in (1..nb_layers).rev() {
        biases[l - 1] = &biases[l - 1] - &(&deltas[l - 1] * learning_rate) - epsilon;

        for j in 0..deltas[l - 1].len() {
            for k in 0..values[l - 1].len() {
                let a = values[l - 1].get(k);
                let d = deltas[l - 1].get(j);
                let w = weights[l - 1].get_at(j, k);
                weights[l - 1].set_at(j, k, w - learning_rate * (a * d) + epsilon);
            }
        }
    }
}
