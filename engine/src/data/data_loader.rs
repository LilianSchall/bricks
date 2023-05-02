use std::fs;
use crate::maths::Matrix;

pub fn load_data(path: &str) -> Vec<(Matrix, Matrix)> {
    let contents = fs::read_to_string(path).expect("Loading path is invalid");

    let lines = contents.split("\n").collect::<Vec<&str>>();
    let mut res: Vec<(Matrix, Matrix)> = vec![];

    for i in (0..lines.len() - 1).step_by(2) {
        let input = create_vec(lines[i]);
        let i_length = input.len();
        let output = create_vec(lines[i+1]);
        let o_length = output.len();

        res.push((Matrix::reshape(input, 1, i_length).unwrap(),
                  Matrix::reshape(output, 1, o_length).unwrap()));
    }
    res
}

fn create_vec(string: &str) -> Vec<f64> {
    string.split(" ")
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<f64>>()
}