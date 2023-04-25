use bricks::activations::dense_activation::DenseActivation;
use bricks::losses::losses::Loss;
use bricks::maths::matrix::Matrix;
use bricks::models::dense_model;
use bricks::models::dense_model::DenseModel;
use bricks::sessions::session;
use bricks::sessions::session::Session;
use bricks::shapes::dense_shape::DenseShape;

fn main() {

    let activations = vec![DenseActivation::Relu, DenseActivation::Tanh];
    let shape = vec![DenseShape::new(2,1,1), DenseShape::new(3,1,1), DenseShape::new(1,1,1)];
    let mut model = DenseModel::new(activations, Loss::MeanSquaredError, shape, None);

    let training_data = vec![
        (Matrix::from(vec![1.0,0.0]), Matrix::from(vec![1.0])),
        (Matrix::from(vec![0.0,1.0]), Matrix::from(vec![1.0])),
        (Matrix::from(vec![1.0,1.0]), Matrix::from(vec![0.0])),
        (Matrix::from(vec![0.0,0.0]), Matrix::from(vec![0.0]))];
    let testing_data = training_data.clone();

    let mut session = Session::new(model, 1E-2, training_data, testing_data, 50000, Some(0.005));

    session.fit();
    model = session.give_model();
}
