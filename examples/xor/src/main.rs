use bricks::activations::DenseActivation;
use bricks::data::load_data;
use bricks::losses::Loss;
use bricks::networks::{DenseNetwork, Network};
use bricks::sessions::Session;
use bricks::shapes::DenseShape;

fn main() {

    let mut network : DenseNetwork;
    let save_exist = std::path::Path::new("xor.save").exists();
    if !save_exist {
        println!("Creating network");
        let activations = vec![DenseActivation::Relu, DenseActivation::Tanh];
        let shape = vec![DenseShape::one_d(2), DenseShape::one_d(3), DenseShape::one_d(1)];
        network = DenseNetwork::new(activations, Loss::MeanSquaredError, shape, None);
    }
    else {
        println!("Loading network from save");
        network = DenseNetwork::load_network("xor.save");
    }

    let training_data = load_data("training_data.dat");

    let testing_data = training_data.clone();

    let mut session = Session::new(network, 1E-2, training_data, testing_data, 50000, Some(0.005), true);

    println!("Error value: {}", if !save_exist {session.fit()} else {session.test()});
    network = session.give_network();
    network.save_network("xor.save");
}
