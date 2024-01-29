use crate::abstractions::ActivatiorDeactivatior;
use rand::rngs::OsRng;
use rand::RngCore;

#[derive(Clone, Debug)]
pub enum MatrixError {
    OutsideRange,
}

/// Matrix holds values in a matix of specified number of rows and columns.
///
#[derive(Clone, Debug)]
pub struct Matrix<T: ActivatiorDeactivatior + Copy> {
    rows: usize,
    cols: usize,
    act_deact: T,
    values: Vec<f64>,
}

impl<T: ActivatiorDeactivatior + Copy> Matrix<T> {
    /// new creates an instance of a new Matrix
    ///
    pub fn new(rows: usize, cols: usize, act_deact: T) -> Matrix<T> {
        return Matrix {
            rows: rows,
            cols: cols,
            act_deact: act_deact,
            values: vec![0.0; (rows * cols) as usize],
        };
    }

    /// get_rows_num returns number of rows
    ///
    pub fn get_rows_num(&self) -> usize {
        self.rows
    }

    /// get_cols_num returns number of columns
    ///
    pub fn get_cols_num(&self) -> usize {
        self.cols
    }

    /// get_at returns value at given row and col
    ///
    pub fn get_at(&self, row: usize, col: usize) -> Result<f64, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutsideRange);
        }

        Ok(self.values[row * self.cols + col])
    }

    /// randomize randomizes all values in the Matrix
    ///
    pub fn randomize(&mut self) {
        let mut rng = OsRng;
        for value in self.values.iter_mut() {
            *value = rng.next_u64() as f64 / u64::MAX as f64;
        }
    }

    /// activate activates values at given row and column
    ///
    pub fn activate(&mut self, row: usize, col: usize) -> Result<f64, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutsideRange);
        }
        let idx: usize = row * self.cols + col;

        self.values[idx] = self.act_deact.act_f(self.values[idx]);

        Ok(self.values[idx])
    }

    /// print prints Matrix to stdout
    pub fn print(&self) {
        println!("matrix = [");
        for i in 0..self.rows {
            print!("  ");
            for j in 0..self.cols {
                print!("{:.3}, ", self.values[i * self.cols + j]);
            }
            print!("\n");
        }
        println!("  ]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activators::Sigmoid;

    #[test]
    fn test_create_matrix_and_getters() {
        let (rows, cols): (usize, usize) = (10, 20);
        let m = Matrix::new(rows, cols, Sigmoid::new());
        assert_eq!(m.get_rows_num(), rows);
        assert_eq!(m.get_cols_num(), cols);
    }

    #[test]
    fn test_randomize_matrix() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.print();
        m.randomize();
        m.print();
    }

    #[test]
    fn test_activate_matrix_value() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        let before_act: f64 = m.get_at(5, 5).unwrap();
        let _ = m.activate(5, 5);
        let after_act: f64 = m.get_at(5, 5).unwrap();
        assert_ne!(before_act, after_act);
    }
}
