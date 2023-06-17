use bricks::activations::DenseActivation;
use bricks::data::{load_data, split_data};
use bricks::losses::Loss;
use bricks::networks::{DenseNetwork, Network};
use bricks::sessions::{DenseSession, Session};
use bricks::shapes::DenseShape;

pub fn train_network() {
    let activations = vec![
        DenseActivation::Sigmoid,
        DenseActivation::Sigmoid,
        DenseActivation::Sigmoid,
        DenseActivation::Softmax,
    ];
    let shape = vec![
        DenseShape::new(28, 28, 1),
        DenseShape::one_d(16),
        DenseShape::one_d(16),
        DenseShape::one_d(16),
        DenseShape::one_d(10),
    ];


    let mut network = DenseNetwork::new(
        activations,
        Loss::CrossEntropy,
        shape,
        None,
    );
    let data = load_data("small_data.dat");
    let (training_data, testing_data) = split_data(data, 30);


    println!("Data loaded!");
    let mut session = DenseSession::new(
        network,
        1E-1,
        training_data,
        testing_data,
        50,
        Some(0.005),
        true,
        Some(100)
    );

    println!("Launching session fitting!");
    session.fit();
    network = session.release_network();
    network.save_network("digit_reader.save");
}
