use criterion::{criterion_group, criterion_main, Criterion};

use neural_cluster::activators::ActivatorOption;
use neural_cluster::matrix::Matrix;
use neural_cluster::nn::{LayerSchema, NN};

fn nn_cost_benchmark(c: &mut Criterion) {
    c.bench_function("nn_cost_benchmark", |b| {
        let schema: Vec<LayerSchema> = vec![
            LayerSchema {
                size: 100,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 100,
                activator: ActivatorOption::Tanh,
                alpha: 0.0,
            },
            LayerSchema {
                size: 100,
                activator: ActivatorOption::ReLU,
                alpha: 0.0,
            },
        ];
        let nn = NN::new(&schema);
        let mut nnn = nn.unwrap();
        nnn.randomize();

        let mut input = Matrix::new(1000, 100);
        input.randomize();

        let mut output = Matrix::new(1000, 100);
        output.randomize();

        b.iter(|| {
            let _ = nnn.cost(&input, &output);
        })
    });
}

fn nn_backprop_benchmark(c: &mut Criterion) {
    c.bench_function("nn_cost_benchmark", |b| {
        let schema: Vec<LayerSchema> = vec![
            LayerSchema {
                size: 100,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 100,
                activator: ActivatorOption::Tanh,
                alpha: 0.0,
            },
            LayerSchema {
                size: 100,
                activator: ActivatorOption::ReLU,
                alpha: 0.0,
            },
        ];
        let nn = NN::new(&schema);
        let mut nnn = nn.unwrap();
        nnn.randomize();

        let mut input = Matrix::new(1000, 100);
        input.randomize();

        let mut output = Matrix::new(1000, 100);
        output.randomize();
        let mut mem = nnn.create_mem();

        b.iter(|| {
            if let Err(err) = nnn.backprop(&mut mem, &input, &output) {
                panic!("error: {:?}", err);
            }
        })
    });
}



criterion_group!(benches, nn_cost_benchmark, nn_backprop_benchmark);
criterion_main!(benches);
