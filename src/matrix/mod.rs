use crate::abstractions::ActivatiorDeactivatior;
use rand::rngs::OsRng;
use rand::RngCore;

#[derive(Clone, Debug, PartialEq)]
pub enum MatrixError {
    OutsideRange,
}

/// Matrix holds values in a matrix of specified number of rows and columns.
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
        self.values
            .iter_mut()
            .for_each(|v: &mut f64| *v = rng.next_u64() as f64 / u64::MAX as f64)
    }

    /// activatel activates all values
    ///
    pub fn activate(&mut self) {
        self.values
            .iter_mut()
            .for_each(|v: &mut f64| self.act_deact.act_f(v));
    }
    pub fn de_activate(&mut self) {
        self.values
            .iter_mut()
            .for_each(|v: &mut f64| self.act_deact.de_act_f(v));
    }

    /// print prints Matrix to stdout
    ///
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

    /// normalize normalizes the value between min and max
    ///
    pub fn normalize(&mut self, min: f64, max: f64) {
        self.values
            .iter_mut()
            .for_each(|x: &mut f64| *x = (*x - min) / (max - min));
    }

    pub fn unormalize(&mut self, min: f64, max: f64) {
        self.values.iter_mut().for_each(|x: &mut f64| {
            *x = *x * (max - min) + min;
        })
    }

    /// min_max returns min and max values
    ///
    pub fn min_max(&self) -> (f64, f64) {
        if self.values.len() == 0 {
            return (0.0, 0.0);
        }
        let mut min: f64 = f64::MAX;
        let mut max: f64 = f64::MIN;
        self.values.iter().for_each(|x: &f64| {
            if *x > max {
                max = *x;
            }
            if *x < min {
                min = *x;
            }
        });
        (min, max)
    }

    /// compare compares all the matrix values with other matrix values
    ///
    pub fn compare(&self, other: &Matrix<T>, f: fn((&f64, &f64)) -> bool) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        if self.values.len() != other.values.len() {
            return false;
        }
        self.values.iter().zip(other.values.iter()).all(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activators::Sigmoid;
    use crate::float_math::round;

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
        m.randomize();
        m.values
            .iter()
            .for_each(|v: &f64| assert_eq!(true, *v > 0.0 && *v < 1.0));
    }

    #[test]
    fn test_get_at() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        for r in 0..m.get_rows_num() {
            for c in 0..m.get_cols_num() {
                let v = m.get_at(r, c);
                match v {
                    Ok(v) => assert_ne!(v, 0.0),
                    Err(err) => panic!("{:?}", err),
                }
            }
        }
        let v = m.get_at(11, 20);
        match v {
            Ok(v) => panic!("unreachable"),
            Err(err) => assert_eq!(err, MatrixError::OutsideRange),
        }
        let v = m.get_at(10, 21);
        match v {
            Ok(v) => panic!("unreachable"),
            Err(err) => assert_eq!(err, MatrixError::OutsideRange),
        }
    }

    #[test]
    fn test_activate_matrix_value() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        let cp_m = m.clone();
        m.activate();
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_de_activate_matrix_value() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        let cp_m = m.clone();
        m.de_activate();
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_normalize_unarmalize_values() {
        let (rows, cols): (usize, usize) = (10, 10);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        let nom_m = m.clone();
        m.unormalize(0.0, 100.0);
        m.normalize(0.0, 100.0);
        m.values
            .iter()
            .zip(nom_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(round(*x, 6), round(*y, 6));
            })
    }

    #[test]
    fn test_min_max() {
        let m = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0,
                3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0,
                9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0,
                100.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0,
            ],
            act_deact: Sigmoid::new(),
        };
        let (min, max) = m.min_max();
        assert_eq!(min, 0.0);
        assert_eq!(max, 100.0);
    }
}
