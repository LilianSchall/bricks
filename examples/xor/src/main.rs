use bricks::activations::DenseActivation;
use bricks::data::load_data;
use bricks::losses::Loss;
use bricks::maths::Matrix;
use bricks::models::DenseModel;
use bricks::sessions::Session;
use bricks::shapes::DenseShape;

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

    let training_data = load_data("training_data.dat");

    let testing_data = training_data.clone();

    let mut session = Session::new(model, 1E-2, training_data, testing_data, 50000, Some(0.005), true);

    println!("Error value: {}", if !save_exist {session.fit()} else {session.test()});
    model = session.give_model();
    model.save_model("xor.save");
}
