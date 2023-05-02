use crate::activations::DenseActivation;
use crate::maths::Matrix;

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