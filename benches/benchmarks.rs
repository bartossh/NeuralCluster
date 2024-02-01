use criterion::{criterion_group, criterion_main, Criterion};

use neural_cluster::activators::ActivatorOption;
use neural_cluster::matrix::Matrix;
use neural_cluster::nn::{LayerSchema, NN};

fn nn_cost_benchmark(c: &mut Criterion) {
    c.bench_function("nn_cost_benchmark", |b| {
        let schema: Vec<LayerSchema> = vec![
            LayerSchema {
                size: 10,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 10,
                activator: ActivatorOption::Tanh,
                alpha: 0.0,
            },
            LayerSchema {
                size: 10,
                activator: ActivatorOption::ReLU,
                alpha: 0.0,
            },
        ];
        let nn = NN::new(&schema);
        let mut nnn = nn.unwrap();
        nnn.randomize();

        let mut input = Matrix::new(10, 10);
        input.randomize();

        let mut output = Matrix::new(10, 10);
        output.randomize();

        b.iter(|| {
            let _ = nnn.cost(&input, &output);
        })
    });
}

criterion_group!(benches, nn_cost_benchmark);
criterion_main!(benches);
