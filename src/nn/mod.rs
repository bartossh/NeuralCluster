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
    /// Prints neural network layers activations, weights and bias matrices.
    ///
    pub fn print(&self) {
        println!("Neural network:");
        for l in 0..self.layers.len() { 
            let activations = &self.layers[l].activations;
            let weights = &self.layers[l].weights;
            let bias = &self.layers[l].bias;
            println!(" --- layer {} ---", l);
            activations.borrow().print();
            if let (Some(weights), Some(bias)) = (weights, bias) {
                weights.borrow().print();
                bias.borrow().print();
            }
        }
    }

    /// Randomizes all the activations, bies and weigths layser.
    ///
    pub fn randomize(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.activations.borrow_mut().randomize();
            
            let weights = &l.weights;
            let bias = &l.bias;
            if let (Some(weights), Some(bias)) = (weights, bias) {
                weights.borrow_mut().randomize();
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
        if cost.is_nan() {
            return Err(NNError::Fatal);
        }

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
            let mut new_layer = Layer {
                activations: Matrix::new(activations.get_rows_num(), activations.get_cols_num()).into(),
                weights: None,
                bias: None,
                activator: None,
            };
            if let Some(w) = &l.weights {
                let w = w.borrow();
                new_layer.weights = Some(Matrix::new(w.get_rows_num(), w.get_cols_num()).into());
            }
            
            if let Some(b) = &l.bias {
                let b = b.borrow();
                new_layer.bias = Some(Matrix::new(b.get_rows_num(), b.get_cols_num()).into());
            }
            mem.layers.push(new_layer);
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
                    if let Err(err) = mem.layers[mem.layers.len()-1].activations.borrow_mut().set_at(0, c, so - oo) {
                        return Err(NNError::Fatal);
                    }
                } else {
                    return Err(NNError::Fatal);
                }
            }

            for l in (1..self.layers.len()).rev() {
                for c in 0..self.layers[l].activations.borrow().get_cols_num() {
                    let sa = self.layers[l].activations.borrow().get_at(0, c);
                    let ma = mem.layers[l].activations.borrow().get_at(0, c);

                    if let (Ok(mut sa), Ok(ma)) = (sa, ma) {
                        if let Some(activator) = &self.layers[l].activator {
                            activator.de_act_f(&mut sa);
                            let mbi = &mem.layers[l-1].bias;
                            if let Some(mbi) = mbi {
                                let _ = mbi.borrow_mut().add_at(0, c, 2.0*ma*sa);
                            } else {
                                return Err(NNError::Fatal);
                            }
                        }

                        for c_p in 0..self.layers[l-1].activations.borrow().get_cols_num() {
                            let sa_p = self.layers[l-1].activations.borrow().get_at(0, c_p);
                            let sw_p = match &self.layers[l-1].weights {
                                Some(w) =>  w.borrow().get_at(c_p, c),
                                None => Err(MatrixError::Fatal),
                            };

                            if let (Ok(sa_p), Ok(sw_p)) = (sa_p, sw_p) {
                                let wm = &mem.layers[l-1].weights;
                                if let Some(wm) = wm {
                                    if let Err(err) = wm.borrow_mut().add_at(c_p, c, 2.0*ma*sa*sa_p){
                                        return Err(NNError::Fatal);
                                    }
                                } else {
                                    return Err(NNError::Fatal);
                                }
                                if let Err(_) = mem.layers[l-1].activations.borrow_mut().set_at(0, c_p, 2.0*ma*sa*sw_p) {
                                    return Err(NNError::Fatal);
                                }
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
                let rows = weights.borrow().get_rows_num();
                let cols = weights.borrow().get_cols_num();
                for r in 0..rows {
                    for c in 0.. cols {
                        let mut wv_update = 0.0;
                        if let Ok(wv) = weights.borrow().get_at(r, c) {
                            wv_update = wv/(rows as f64);
                        } else {
                            return Err(NNError::Fatal);
                        }
                        if let Err(_) = weights.borrow_mut().set_at(r, c, wv_update) {
                            return Err(NNError::Fatal);
                        }
                    }
                }
            }
            if let Some(bias) = &mem.layers[l].bias {
                let cols = bias.borrow().get_cols_num();
                for c in 0..cols {
                    let mut bv_update = 0.0;
                    if let Ok(bv) = bias.borrow().get_at(0, c) {
                        bv_update = bv/(rows as f64); 
                    } else {
                        return Err(NNError::Fatal);
                    }
                    if let Err(_) = bias.borrow_mut().set_at(0, c, bv_update) {
                        return Err(NNError::Fatal);
                    }
                }
            }
        }

        Ok(())
    }

    /// Performs learning process by applying corrections with given learning rate 
    /// from mem neural network. Function assumes that mem has the same architecture
    /// and the same matrices sizes for all layers. Returns NNError if any of above isn't met.
    /// The mem and self architecture isn't validated beforehand for performance purposes.
    ///
    pub fn learn(&mut self, mem: &NN, rate: f64) -> Result<(), NNError> {
        if self.layers.len() != self.layers.len() {
            return Err(NNError::WrongSchemaLength);
        }
        for l in 0..self.layers.len()-1 {
            if let (Some(s_weights), Some(m_weights)) = (&self.layers[l].weights, &mem.layers[l].weights) {
                let rows = s_weights.borrow().get_rows_num();
                let cols = s_weights.borrow().get_cols_num();
                for r in 0..rows {
                    for c in 0..cols {
                        let mut mw_v: f64 = 0.0;
                        if let Ok(mw) = m_weights.borrow().get_at(r, c) {
                            mw_v = mw;
                        } else {
                            return Err(NNError::Fatal);
                        }
                        if let Err(_) = s_weights.borrow_mut().substract_at(r, c, mw_v*rate){
                            return Err(NNError::Fatal);
                        }
                    }
                }
            }
            if let (Some(s_bias), Some(m_bias)) = (&self.layers[l].bias, &mem.layers[l].bias) {
                let cols = s_bias.borrow().get_cols_num();
                for c in 0..cols {
                    let mut mb_v: f64 = 0.0;
                    if let Ok(mb) = m_bias.borrow().get_at(0, c) {
                        mb_v = mb;
                    } else {
                        return Err(NNError::Fatal);
                    }
                    if let Err(_) = s_bias.borrow_mut().substract_at(0, c, mb_v*rate){
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
                size: 6,
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
                size: 6,
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
                activator: ActivatorOption::Sigmoid,
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
    
    #[test]
    fn test_nn_learing_simplistic_or_problem() {
        let schema: Vec<LayerSchema> = vec![
            LayerSchema {
                size: 2,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 1,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 1,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
        ];
        let nn = NN::new(&schema);
        let mut nnn = nn.unwrap();
        nnn.randomize();

        let in_out_rows = 4;
        let mut input = Matrix::new(in_out_rows, 2);
        let mut output = Matrix::new(in_out_rows, 1);
        
        let _ = input.set_at(0, 0, 0.0);
        let _ = input.set_at(0, 1, 0.0);
        let _ = output.set_at(0, 0, 0.0);
        let _ = input.set_at(1, 0, 0.0);
        let _ = input.set_at(1, 1, 1.0);
        let _ = output.set_at(1, 0, 1.0);
        let _ = input.set_at(2, 0, 1.0);
        let _ = input.set_at(2, 1, 0.0);
        let _ = output.set_at(2, 0, 1.0);
        let _ = input.set_at(3, 0, 1.0);
        let _ = input.set_at(3, 1, 1.0);
        let _ = output.set_at(3, 0, 1.0);

        let mut mem = nnn.create_mem();
        let epochs: usize = 1000000;
        let learning_rate: f64 = 0.1;
        let mut found: usize = 0;
        for _ in 0..epochs {
            if let Err(err) = nnn.backprop(&mut mem, &input, &output) {
                panic!("error: {:?}", err);
            }

            if let Err(err) = nnn.learn(&mem, learning_rate) {
                panic!("error: {:?}", err);
            }
            
            found = 0;
            for i in 0..in_out_rows {
                let in_row = &input.get_row(i);
                let out_row = &output.get_row(i);
                if let (Ok(in_row), Ok(out_row)) = (in_row, out_row) {
                    
                    if let Err(err) = nnn.input(in_row) {
                        panic!("error: {:?}", err);
                    }
                    
                    if let Err(err) = nnn.forward() {
                        panic!("error: {:?}", err);
                    }

                    let mut output = Matrix::new(1,1);

                    if let Err(err) = nnn.output(&mut output) {
                        panic!("error: {:?}", err);
                    }

                    if let (Ok(test_value), Ok(calc_value)) = (&out_row.get_at(0, 0), &output.get_at(0,0)) {
                        if (*test_value - *calc_value).abs() < 0.1 {
                            found+=1;
                        }
                    } 
                }
                if found == in_out_rows {
                    break;
                }
            }
        }
        for i in 0..in_out_rows {
            let in_row = &input.get_row(i);
            let out_row = &output.get_row(i);
            if let (Ok(in_row), Ok(out_row)) = (in_row, out_row) {
                if let Err(err) = nnn.input(in_row) {
                    panic!("error: {:?}", err);
                }
                
                if let Err(err) = nnn.forward() {
                    panic!("error: {:?}", err);
                }

                let mut output = Matrix::new(1,1);

                if let Err(err) = nnn.output(&mut output) {
                    panic!("error: {:?}", err);
                } 

                if let (Ok(test_value), Ok(calc_value)) = (&out_row.get_at(0, 0), &output.get_at(0,0)) {
                    println!("expected: [ {:.3} ] | got [ {:.3} ]", *test_value, calc_value);
                } 
            }
        }
        assert_eq!(found, in_out_rows);
    }

    //#[test]
    fn test_nn_learing_simplistic_xor_problem() {
        let schema: Vec<LayerSchema> = vec![
            LayerSchema {
                size: 2,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 4,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
            LayerSchema {
                size: 1,
                activator: ActivatorOption::Sigmoid,
                alpha: 0.0,
            },
        ];
        let nn = NN::new(&schema);
        let mut nnn = nn.unwrap();
        nnn.randomize();

        let in_out_rows = 4;
        let mut input = Matrix::new(in_out_rows, 2);
        let mut output = Matrix::new(in_out_rows, 1);
        
        let _ = input.set_at(0, 0, 0.0);
        let _ = input.set_at(0, 1, 0.0);
        let _ = output.set_at(0, 0, 0.0);
        let _ = input.set_at(1, 0, 0.0);
        let _ = input.set_at(1, 1, 1.0);
        let _ = output.set_at(1, 0, 1.0);
        let _ = input.set_at(2, 0, 1.0);
        let _ = input.set_at(2, 1, 0.0);
        let _ = output.set_at(2, 0, 1.0);
        let _ = input.set_at(3, 0, 1.0);
        let _ = input.set_at(3, 1, 1.0);
        let _ = output.set_at(3, 0, 0.0);

        let mut mem = nnn.create_mem();
        let epochs: usize = 1000000;
        let learning_rate: f64 = 0.1;
        let mut found: usize = 0;
        for _ in 0..epochs { 
            if let Err(err) = nnn.backprop(&mut mem, &input, &output) {
                panic!("error: {:?}", err);
            }

            if let Err(err) = nnn.learn(&mem, learning_rate) {
                panic!("error: {:?}", err);
            }

            found = 0;
            for i in 0..in_out_rows {
                let in_row = &input.get_row(i);
                let out_row = &output.get_row(i);
                if let (Ok(in_row), Ok(out_row)) = (in_row, out_row) {
                    if let Err(err) = nnn.input(in_row) {
                        panic!("error: {:?}", err);
                    }
                    
                    if let Err(err) = nnn.forward() {
                        panic!("error: {:?}", err);
                    }

                    let mut output = Matrix::new(1,1);

                    if let Err(err) = nnn.output(&mut output) {
                        panic!("error: {:?}", err);
                    } 

                    if let (Ok(test_value), Ok(calc_value)) = (&out_row.get_at(0, 0), &output.get_at(0,0)) {
                        if (*test_value - *calc_value).abs() < 0.1 {
                            found+=1;
                        }
                    } 
                }
            }
            if found == in_out_rows {
                break;
            }
        }
        for i in 0..in_out_rows {
            let in_row = &input.get_row(i);
            let out_row = &output.get_row(i);
            if let (Ok(in_row), Ok(out_row)) = (in_row, out_row) {
                if let Err(err) = nnn.input(in_row) {
                    panic!("error: {:?}", err);
                }
                
                if let Err(err) = nnn.forward() {
                    panic!("error: {:?}", err);
                }

                let mut output = Matrix::new(1,1);

                if let Err(err) = nnn.output(&mut output) {
                    panic!("error: {:?}", err);
                } 

                if let (Ok(test_value), Ok(calc_value)) = (&out_row.get_at(0, 0), &output.get_at(0,0)) {
                    println!("expected: [ {:.3} ] | got [ {:.3} ]", *test_value, calc_value);
                } 
            }
        }

        assert_eq!(found, in_out_rows);
    }
}
