use crate::activators::{ActivatorDeactivator, ActivatorOption, LeakyReLU, ReLU, Sigmoid, Tanh};
use crate::matrix::{Matrix, MatrixError};

/// NN crate errors.
///
#[derive(Clone, Debug, PartialEq)]
pub enum NNError {
    WrongSchemaLength,
}

/// Describes the Layer schema.
///
#[derive(Clone, Debug)]
struct LayerSchema {
    size: usize,
    activator: ActivatorOption,
    alpha: f64,
}

/// Contains activations, weights and bias matrices with perceptron activator and deactivator
/// and all other functionalities to be applied to that layer
///
#[derive(Debug)]
struct Layer {
    activations: Matrix,
    bias: Option<Matrix>,
    weights: Option<Matrix>,
    activator: Option<Box<dyn ActivatorDeactivator>>,
}

/// Neural network.
///
#[derive(Debug)]
pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    /// Creates a new instance of NN.
    ///
    pub fn new(schema: &Vec<LayerSchema>) -> Result<NN, NNError> {
        if schema.len() < 3 {
            return Err(NNError::WrongSchemaLength);
        }
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..schema.len() - 1 {
            layers.push(Layer {
                activations: Matrix::new(1, schema[i].size),
                bias: Some(Matrix::new(1, schema[i + 1].size)),
                weights: Some(Matrix::new(schema[i].size, schema[i + 1].size)),
                activator: schema[i].activator.get_activator(schema[i].alpha),
            });
        }
        layers.push(Layer {
            activations: Matrix::new(1, schema[0].size),
            bias: None,
            weights: None,
            activator: None,
        });

        Ok(NN { layers: layers })
    }

    /// Randomizes all the activations, bies and weigths layser.
    ///
    pub fn randomize(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.activations.randomize();
            if let Some(ref mut bias) = l.bias {
                bias.randomize();
            }
            if let Some(ref mut bias) = l.bias {
                bias.randomize();
            }
        });
    }

    /// Copies values from input matrix in to the first self layer.
    ///
    pub fn input(&mut self, input: &Matrix) -> Result<(), MatrixError> {
        self.layers[0].activations.copy_to_self(input)
    }

    /// Copies values from the self output layer to the output matrix.
    ///
    pub fn output(&self, output: &mut Matrix) -> Result<(), MatrixError> {
        output.copy_to_self(&self.layers[self.layers.len() - 1].activations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activators::ActivatorOption;

    #[test]
    fn test_create_new_nn() {
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
        match nn {
            Ok(nn) => assert_eq!(nn.layers.len(), 3),
            Err(err) => panic!("error: {:?}", err),
        };
    }

    #[test]
    fn test_randomize_nn() {
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

        if let Ok(value) = nnn.layers[2].activations.get_at(0, 0) {
            assert_ne!(value, 0.0);
        }
    }

    #[test]
    fn test_input_nn() {
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

        let mut input = Matrix::new(1, 10);
        input.randomize();

        if let Err(err) = nnn.input(&input) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            nnn.layers[0]
                .activations
                .compare(&input, |(x, y): (&f64, &f64)| x == y)
        )
    }

    #[test]
    fn test_output_nn() {
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

        let mut output = Matrix::new(1, 10);

        if let Err(err) = nnn.output(&mut output) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            nnn.layers[nnn.layers.len() - 1]
                .activations
                .compare(&output, |(x, y): (&f64, &f64)| x == y)
        )
    }
}
