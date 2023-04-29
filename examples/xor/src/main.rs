use bricks::activations::dense_activation::DenseActivation;
use bricks::losses::losses::Loss;
use bricks::maths::matrix::Matrix;
use bricks::models::dense_model::DenseModel;
use bricks::sessions::session::Session;
use bricks::shapes::dense_shape::DenseShape;

fn main() {

    let mut model : DenseModel;
    let save_exist = std::path::Path::new("xor.save").exists();
    if !save_exist {
        println!("Creating model");
        let activations = vec![DenseActivation::Relu, DenseActivation::Tanh];
        let shape = vec![DenseShape::one_d(2), DenseShape::one_d(3), DenseShape::one_d(1)];
        model = DenseModel::new(activations, Loss::MeanSquaredError, shape, None);
    }
    else {
        println!("Loading model from save");
        model = DenseModel::load_model("xor.save");
    }

    let training_data = vec![
        (Matrix::from(vec![1.0,0.0]), Matrix::from(vec![1.0])),
        (Matrix::from(vec![0.0,1.0]), Matrix::from(vec![1.0])),
        (Matrix::from(vec![1.0,1.0]), Matrix::from(vec![0.0])),
        (Matrix::from(vec![0.0,0.0]), Matrix::from(vec![0.0]))];
    let testing_data = training_data.clone();

    let mut session = Session::new(model, 1E-2, training_data, testing_data, 50000, Some(0.005), true);

    println!("Error value: {}", if !save_exist {session.fit()} else {session.test()});
    model = session.give_model();
    model.save_model("xor.save");
}
