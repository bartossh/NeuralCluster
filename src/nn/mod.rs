use crate::activators::{ActivatorDeactivator, ActivatorOption};
use crate::matrix::{Matrix, MatrixError};
use std::cell::RefCell;

/// NN crate errors.
///
#[derive(Clone, Debug, PartialEq)]
pub enum NNError {
    WrongSchemaLength,
}

/// Describes the Layer schema.
///
#[derive(Clone, Debug)]
pub struct LayerSchema {
    size: usize,
    activator: ActivatorOption,
    alpha: f64,
}

/// Contains activations, weights and bias matrices with perceptron activator and deactivator
/// and all other functionalities to be applied to that layer
///
#[derive(Debug)]
struct Layer {
    activations: RefCell<Matrix>,
    bias: Option<RefCell<Matrix>>,
    weights: Option<RefCell<Matrix>>,
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
                activations: Matrix::new(1, schema[i].size).into(),
                bias: Some(Matrix::new(1, schema[i + 1].size).into()),
                weights: Some(Matrix::new(schema[i].size, schema[i + 1].size).into()),
                activator: if i != 0 {
                    schema[i].activator.get_activator(schema[i].alpha)
                } else {
                    None
                },
            });
        }
        layers.push(Layer {
            activations: Matrix::new(1, schema[schema.len() - 1].size).into(),
            bias: None,
            weights: None,
            activator: schema[schema.len() - 1]
                .activator
                .get_activator(schema[schema.len() - 1].alpha),
        });

        Ok(NN { layers: layers })
    }

    /// Randomizes all the activations, bies and weigths layser.
    ///
    pub fn randomize(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.activations.borrow_mut().randomize();
            if let Some(ref mut bias) = l.bias {
                bias.borrow_mut().randomize();
            }
            if let Some(ref mut bias) = l.bias {
                bias.borrow_mut().randomize();
            }
        });
    }

    /// Copies values from input matrix in to the first self layer.
    ///
    pub fn input(&mut self, input: &Matrix) -> Result<(), MatrixError> {
        self.layers[0].activations.borrow_mut().copy_to_self(input)
    }

    /// Copies values from the self output layer to the output matrix.
    ///
    pub fn output(&self, output: &mut Matrix) -> Result<(), MatrixError> {
        output.copy_to_self(&self.layers[self.layers.len() - 1].activations.borrow())
    }

    /// Runs feed forward activations for NN.
    ///
    pub fn forward(&mut self) -> Result<(), MatrixError> {
        for i in 0..self.layers.len() - 1 {
            let mut rec = self.layers[i + 1].activations.borrow_mut();
            let act = self.layers[i].activations.borrow();
            if let Some(weights) = &self.layers[i].weights {
                let wgh = weights.borrow();
                if let Err(err) = rec.dot(&act, &wgh) {
                    return Err(err);
                }
            }
            if let Some(bias) = &self.layers[i].bias {
                let b = bias.borrow();
                if let Err(err) = rec.sum(&b) {
                    return Err(err);
                }
            }
            if let Some(activator) = &self.layers[i].activator {
                let act: &dyn ActivatorDeactivator = &**activator;
                rec.activate(act);
            }
        }
        Ok(())
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
        let activations = nnn.layers[nnn.layers.len() - 1].activations.borrow();

        if let Ok(value) = activations.get_at(0, 0) {
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

        let activations = nnn.layers[0].activations.borrow();

        assert_eq!(
            true,
            activations.compare(&input, |(x, y): (&f64, &f64)| x == y)
        );
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

        let activations = nnn.layers[nnn.layers.len() - 1].activations.borrow();

        assert_eq!(
            true,
            activations.compare(&output, |(x, y): (&f64, &f64)| x == y)
        );
    }

    #[test]
    fn test_forward_nn() {
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

        let mut output = Matrix::new(1, 10);

        if let Err(err) = nnn.output(&mut output) {
            panic!("error: {:?}", err);
        }
        {
            let activations_0 = nnn.layers[&nnn.layers.len() - 1].activations.borrow();
            assert_eq!(
                true,
                activations_0.compare(&output, |(x, y): (&f64, &f64)| x == y)
            );
        }
        if let Err(err) = nnn.forward() {
            panic!("error: {:?}", err);
        }

        let activations_1 = nnn.layers[&nnn.layers.len() - 1].activations.borrow();
        assert_ne!(
            true,
            activations_1.compare(&output, |(x, y): (&f64, &f64)| x == y)
        );
    }
}
