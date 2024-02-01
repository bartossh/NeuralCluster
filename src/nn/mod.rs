use crate::activators::{ActivatorDeactivator, ActivatorOption};
use crate::matrix::{Matrix, MatrixError};
use std::cell::RefCell;

/// NN crate errors.
///
#[derive(Clone, Debug, PartialEq)]
pub enum NNError {
    WrongSchemaLength,
    UnmatchingRowsNum,
    Fatal,
}

/// Describes the Layer schema.
///
#[derive(Clone, Debug)]
pub struct LayerSchema {
    pub size: usize,
    pub activator: ActivatorOption,
    pub alpha: f64,
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

    /// Calculates total cost of the neural network.
    ///
    pub fn cost(&mut self, input: &Matrix, output: &Matrix) -> Result<f64, NNError> {
        if !input.has_same_rows_num(output) {
            return Err(NNError::UnmatchingRowsNum);
        }
        let mut cost: f64 = 0.0;
        for r in 0..input.get_rows_num() {
            let row = input.get_row(r);
            if let Ok(row) = row {
                if let Err(_) = self.input(&row) {
                    return Err(NNError::Fatal);
                }
            } else {
                return Err(NNError::Fatal);
            }

            if let Err(_) = self.forward() {
                return Err(NNError::Fatal);
            }

            for c in 0..output.get_cols_num() {
                let so = self.layers[self.layers.len() - 1]
                    .activations
                    .borrow()
                    .get_at(0, c);
                let oo = output.get_at(r, c);
                if let (Ok(so), Ok(oo)) = (so, oo) {
                    let d = so - oo;
                    cost += d * d;
                }
            }
        }

        cost /= output.get_rows_num() as f64;

        Ok(cost)
    }
    
    fn zero(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.activations.borrow_mut().zero();
            if let Some(ref mut bias) = l.bias {
                bias.borrow_mut().zero();
            }
            if let Some(ref mut bias) = l.bias {
                bias.borrow_mut().zero();
            }
        });

    }

    fn zero_activations_layers(&mut self) {
        self.layers.iter().for_each(|l: &Layer| l.activations.borrow_mut().zero());
    }

    /// Calculates the memory matrix that contains back propagated cost in memory NN.
    /// Return NNError if input and output matrix are not matching in row size.
    ///
    pub fn backprop(&mut self, mem: &mut NN, input: &Matrix, output: &Matrix) -> Result<(), NNError> {
        // TODO: check mem matches self

        if !input.has_same_rows_num(output) {
            return Err(NNError::UnmatchingRowsNum);
        }
        mem.zero();

        for r in 0..input.get_rows_num() {
            let row = input.get_row(r);
            if let Ok(row) = row {
                if let Err(_) = self.input(&row) {
                    return Err(NNError::Fatal);
                }
            } else {
                return Err(NNError::Fatal);
            }

            if let Err(_) = self.forward() {
                return Err(NNError::Fatal);
            }

            mem.zero_activations_layers();

            for c in 0..output.get_cols_num() {
                let so = self.layers[self.layers.len()-1].activations.borrow().get_at(0, c);
                let oo = output.get_at(r, c);
                if let (Ok(so), Ok(oo)) = (so, oo) {
                    mem.layers[mem.layers.len()-1].activations.borrow_mut().set_at(0, c, so - oo); 
                }
            }

            for l in self.layers.len()-1..0 {
                for c in 0..self.layers[l].activations.borrow().get_cols_num() {
                    let sv = self.layers[l].activations.borrow().get_at(0, c);
                    let dv = mem.layers[l].activations.borrow().get_at(0, c);
                    let db = match &mem.layers[l-1].bias {
                        Some(b) => b.borrow().get_at(0, c),
                        None => Err(MatrixError::Falal),
                    };

                    if let (Ok(mut sv), Ok(dv), Ok(mut db)) = (sv, dv, db) {
                        if let Some(activator) = &self.layers[l].activator{
                            activator.de_act_f(&mut sv);
                            db += 2.0*dv*sv;
                            let bi = &mem.layers[l-1].bias;
                            if let Some(bi) = bi {
                                bi.borrow_mut().set_at(0, c, db);
                            }
                        }

                        for c_p in 0..self.layers[l-1].activations.borrow().get_cols_num() {
                            let spv = self.layers[l-1].activations.borrow().get_at(0, c_p);
                            let dpv = mem.layers[l-1].activations.borrow().get_at(0, c_p);
                            let spw = match &self.layers[l-1].weights {
                                Some(w) =>  w.borrow().get_at(c_p, c),
                                None => Err(MatrixError::Falal),
                            };
                            let dpw = match &mem.layers[l-1].weights {
                                Some(w) =>  w.borrow().get_at(c_p, c),
                                None => Err(MatrixError::Falal),
                            };

                            if let (Ok(spv), Ok(dpv), Ok(spw), Ok(dpw)) = (spv, dpv, spw, dpw) {
                                let wm = &mem.layers[l-1].weights;
                                if let Some(wm) = wm {
                                    wm.borrow_mut().set_at(c_p, c, dpw+2.0*dv*sv*spv);
                                }
                                mem.layers[l-1].activations.borrow_mut().set_at(0, c_p, dpv + 2.0*dv*sv*spw);
                            } else {
                                return Err(NNError::Fatal);
                            }

                        }

                    } else {
                        return Err(NNError::Fatal);
                    }
                }
            }
        }
        let rows = output.get_rows_num();

        for l in 0..mem.layers.len()-1 {
            if let Some(weights) = &mem.layers[l].weights {
                for r in 0..weights.borrow().get_rows_num() {
                    for c in 0.. weights.borrow().get_cols_num() {
                        if let Ok(wv) = weights.borrow().get_at(r, c) {
                            weights.borrow_mut().set_at(r, c, wv/(rows as f64));
                        } else {
                            return Err(NNError::Fatal);
                        }
                    }
                }
            }
            if let Some(bias) = &mem.layers[l].bias {
                for c in 0..bias.borrow().get_cols_num() {
                    if let Ok(bv) = bias.borrow().get_at(0, c) {
                        bias.borrow_mut().set_at(0, c, bv/(rows as f64));
                    } else {
                        return Err(NNError::Fatal);
                    }
                }
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

    #[test]
    fn test_cost_nn() {
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

        let cost = nnn.cost(&input, &output);

        if let Ok(c) = cost {
            assert_ne!(c, 0.0);
        } else {
            panic!("cost calculation failed");
        }
    }
}
