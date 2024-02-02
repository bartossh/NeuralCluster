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

    /// Creates memory NN that is used by backpropagation process.
    ///
    pub fn create_mem(&self) -> NN {
        let mut mem: NN = NN{layers: Vec::new()};
        self.layers.iter().for_each(|l: &Layer| {

            let activations = &l.activations.borrow();
            let mut newLayer = Layer {
                activations: Matrix::new(activations.get_rows_num(), activations.get_cols_num()).into(),
                weights: None,
                bias: None,
                activator: None,
            };
            if let Some(w) = &l.weights {
                let w = w.borrow();
                newLayer.weights = Some(Matrix::new(w.get_rows_num(), w.get_cols_num()).into());
            }
            
            if let Some(b) = &l.bias {
                let b = b.borrow();
                newLayer.bias = Some(Matrix::new(b.get_rows_num(), b.get_cols_num()).into());
            }
            mem.layers.push(newLayer);
        });

        return mem;
    }

    /// Checks if two neural networks have the same layout.
    ///
    pub fn has_same_layout(&self, other: &NN) -> bool {
        if self.layers.len() != other.layers.len() {
            return false;
        }

        let mut result: bool = true;
        
        self.layers.iter().zip(other.layers.iter()).
            for_each(|(lx, ly): (&Layer, &Layer)| {
                let asx = &lx.activations.borrow();
                let asy = &ly.activations.borrow();
                if asx.get_cols_num() != asy.get_cols_num() {
                    result = false;
                }
                if asx.get_rows_num() != asy.get_rows_num() {
                    result = false;
                }

                if let (Some(wsx), Some(wsy)) = (&lx.weights, &ly.weights) {
                    let wsx = wsx.borrow();
                    let wsy = wsy.borrow();
                    if wsx.get_cols_num() != wsy.get_cols_num() {
                        result = false;
                    }
                    if wsx.get_rows_num() != wsy.get_rows_num() {
                        result = false;
                    }
                }
                
                if let (Some(bsx), Some(bsy)) = (&lx.bias, &ly.bias) {
                    let bsx = bsx.borrow();
                    let bsy = bsy.borrow();
                    if bsx.get_cols_num() != bsy.get_cols_num() {
                        result = false;
                    }
                    if bsx.get_rows_num() != bsy.get_rows_num() {
                        result = false;
                    }
                }
            });

        return result;
    }

    /// Calculates the memory matrix that contains back propagated cost in memory NN.
    /// Return NNError if input and output matrix are not matching in row size.
    /// Function assumes that mem has the same architecture
    /// and the same matrices sizes for all layers. Returns NNError if any of above isn't met.
    /// The mem and self architecture isn't validated beforehand for performance purposes.
    /// This function may panic.
    ///
    pub fn backprop(&mut self, mem: &mut NN, input: &Matrix, output: &Matrix) -> Result<(), NNError> {
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
                let mut rows = 0;
                let mut cols = 0;
                {
                    let weights = weights.borrow();
                    rows = weights.get_rows_num();
                    cols = weights.get_cols_num();
                }
                for r in 0..rows {
                    for c in 0.. cols {
                        let mut wv_update = 0.0;
                        if let Ok(wv) = weights.borrow().get_at(r, c) {
                            wv_update = wv/(rows as f64);
                        } else {
                            return Err(NNError::Fatal);
                        }
                        weights.borrow_mut().set_at(r, c, wv_update);
                    }
                }
            }
            if let Some(bias) = &mem.layers[l].bias {
                let mut cols = 0; 
                {
                    cols = bias.borrow().get_cols_num()
                }
                for c in 0..cols {
                    let mut bv_update = 0.0;
                    if let Ok(bv) = bias.borrow().get_at(0, c) {
                        bv_update = bv/(rows as f64); 
                    } else {
                        return Err(NNError::Fatal);
                    }
                    bias.borrow_mut().set_at(0, c, bv_update);
                }
            }
        }

        Ok(())
    }

    /// Performs learning process by applying corrections with given learning rate 
    /// from mem neural network. Function assumes that mem has the same architecture
    /// and the same matrices sizes for all layers. Returns NNError if any of above isn't met.
    /// The mem and self architecture isn't validated beforehand for performance purposes.
    /// This function may panic.
    ///
    pub fn learn(&mut self, mem: &NN, rate: f64) -> Result<(), NNError> {
        if self.layers.len() != self.layers.len() {
            return Err(NNError::WrongSchemaLength);
        }
        for l in 0..self.layers.len()-1 {
            if let (Some(s_weights), Some(m_weights)) = (&self.layers[l].weights, &mem.layers[l].weights) {
                let mut rows = 0;
                let mut cols = 0;
                {
                    rows = s_weights.borrow().get_rows_num();
                    cols = s_weights.borrow().get_cols_num();
                }
                for r in 0..rows {
                    for c in 0..cols {
                        let mut swv: f64 = 0.0;
                        if let Ok(sw) = s_weights.borrow().get_at(r, c) {
                            swv = sw;
                        } else {
                            return Err(NNError::Fatal);
                        }
                        let mut mwv: f64 = 0.0;
                        if let Ok(mw) = m_weights.borrow().get_at(r, c) {
                            mwv = mw;
                        } else {
                            return Err(NNError::Fatal);
                        }
                        let _ = s_weights.borrow_mut().set_at(r, c, swv - mwv*rate);
                    }
                }
            }
            if let (Some(s_bias), Some(m_bias)) = (&self.layers[l].bias, &mem.layers[l].bias) {
                let mut cols = 0;
                {
                    cols = s_bias.borrow().get_cols_num();
                }
                for c in 0..cols {
                    let mut sbv: f64 = 0.0;
                    if let Ok(sb) = s_bias.borrow().get_at(0, c) {
                        sbv = sb;
                    } else {
                        return Err(NNError::Fatal);
                    }
                    let mut mbv: f64 = 0.0;
                    if let Ok(mb) = m_bias.borrow().get_at(0, c) {
                        mbv = mb;
                    } else {
                        return Err(NNError::Fatal);
                    }
                    let _ = s_bias.borrow_mut().set_at(0, c, sbv - mbv*rate);
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
    
    #[test]
    fn test_dry_backprop_nn() {
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

        let mut mem = nnn.create_mem();

        if let Err(err) = nnn.backprop(&mut mem, &input, &output) {
            panic!("error: {:?}", err);
        }
    }
    
    #[test]
    fn test_dry_learn_nn() {
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

        let mut mem = nnn.create_mem();

        if let Err(err) = nnn.backprop(&mut mem, &input, &output) {
            panic!("error: {:?}", err);
        }

        let learning_rate: f64 = 0.001;

        if let Err(err) = nnn.learn(&mem, learning_rate) {
            panic!("error: {:?}", err);
        }
    }
}
