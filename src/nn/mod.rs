use crate::activators::{ActivatorDeactivator, ActivatorOption, LeakyReLU, ReLU, Sigmoid, Tanh};
use crate::matrix::Matrix;

/// NNError represents nn crate errors
///
#[derive(Clone, Debug, PartialEq)]
pub enum NNError {
    WrongSchemaLength,
}

/// LayerSchema describes the Layer schema
///
#[derive(Clone, Debug)]
struct LayerSchema {
    size: usize,
    activator: ActivatorOption,
    alpha: f64,
}

/// Layer contains Matrix with perceptron activator and deactivator of that matrix
/// and all other functionalities to be applied to that layer
///
#[derive(Debug)]
struct Layer {
    activations: Matrix,
    bias: Option<Matrix>,
    weights: Option<Matrix>,
    activator: Option<Box<dyn ActivatorDeactivator>>,
}

/// NN holds layers of neural network matrices and matrix perceptron activation logic
///
#[derive(Debug)]
pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    /// new creates a new instance of NN
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
}
