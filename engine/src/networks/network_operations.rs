use crate::activations::DenseActivation;
use crate::losses::Loss;
use crate::maths::Matrix;
use crate::shapes::DenseShape;
use std::str::FromStr;

pub fn feed_forward_generics(values: &mut Vec<Matrix>, raw_values: &mut Vec<Matrix>,
                             activations: &Vec<DenseActivation>, weights: &Vec<Matrix>,
                             biases: &Vec<Matrix>, nb_layers: usize, epsilon: f64) {
    for i in 0..(nb_layers - 1) {
        let mut mat = (&weights[i] * &values[i]).unwrap();
        mat = (&mat + &biases[i]).unwrap();
        raw_values[i + 1] = mat.clone();
        activations[i].apply(&mut mat, epsilon);
        values[i + 1] = mat;
    }
}

pub fn load_network_generics(weights: &mut Vec<Matrix>, biases: &mut Vec<Matrix>,
                             activations: &mut Vec<DenseActivation>, loss: &mut Loss,
                             shape: &mut Vec<DenseShape>, lines: Vec<&str>) {
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
                *shape = line.split(" ")
                    .map(|value| DenseShape::one_d(value.parse::<usize>().unwrap()))
                    .collect::<Vec<_>>();
                *weights = Vec::with_capacity(shape.len() - 1);
                *biases = Vec::with_capacity(shape.len() - 1);
            }
            1 => *activations = line.split(" ")
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
}