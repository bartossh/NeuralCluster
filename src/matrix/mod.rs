use crate::activators::ActivatorDeactivator;
use rand::rngs::OsRng;
use rand::RngCore;

/// Matrix crate errors.
///
#[derive(Clone, Debug, PartialEq)]
pub enum MatrixError {
    OutsideRange,
    FilterToLarge,
    NotMatchingSize,
    Fatal,
}

/// Matrix with f64 values stored in rows, cols order.
///
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Matrix {
    /// Creates an instance of a new Matrix.
    ///
    pub fn new(rows: usize, cols: usize) -> Matrix {
        return Matrix {
            rows: rows,
            cols: cols,
            values: vec![0.0; (rows * cols) as usize],
        };
    }

    /// Returns number of rows.
    ///
    pub fn get_rows_num(&self) -> usize {
        self.rows
    }

    /// Returns number of columns.
    ///
    pub fn get_cols_num(&self) -> usize {
        self.cols
    }

    /// Returns value at given row and column.
    ///
    pub fn get_at(&self, row: usize, col: usize) -> Result<f64, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutsideRange);
        }

        Ok(self.values[row * self.cols + col])
    }

    /// Sets value at given row and column.
    ///
    pub fn set_at(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutsideRange);
        }
        self.values[row * self.cols + col] = value;

        Ok(())
    }

    /// Returns copy of a given row as new Matrix.
    ///
    pub fn get_row(&self, row: usize) -> Result<Matrix, MatrixError> {
        if row > self.rows - 1 {
            return Err(MatrixError::OutsideRange);
        }
        let mut m = Matrix::new(1, self.cols);
        m.values = self.values[row * self.cols..row * self.cols + self.cols].to_vec();

        Ok(m)
    }

    /// Randomizes all values.
    ///
    pub fn randomize(&mut self) {
        let mut rng = OsRng;
        self.values
            .iter_mut()
            .for_each(|v: &mut f64| *v = rng.next_u64() as f64 / u64::MAX as f64)
    }

    /// Activates all values.
    ///
    pub fn activate(&mut self, a: &dyn ActivatorDeactivator) {
        self.values.iter_mut().for_each(|v: &mut f64| a.act_f(v));
    }
    pub fn de_activate(&mut self, d: &dyn ActivatorDeactivator) {
        self.values.iter_mut().for_each(|v: &mut f64| d.de_act_f(v));
    }

    /// Prints values in row - column order.
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

    /// Normalizes all values between min and max.
    ///
    pub fn normalize(&mut self, min: f64, max: f64) {
        self.values
            .iter_mut()
            .for_each(|x: &mut f64| *x = (*x - min) / (max - min));
    }

    /// Unormalizes values between min and max.
    ///
    pub fn unormalize(&mut self, min: f64, max: f64) {
        self.values.iter_mut().for_each(|x: &mut f64| {
            *x = *x * (max - min) + min;
        })
    }

    /// Returns min and max values.
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

    /// Checks if both matrices has the same number of rows.
    ///
    pub fn has_same_rows_num(&self, other: &Matrix) -> bool {
        self.rows == other.rows
    }

    /// Compares the size and all the self values with other matrix values.
    ///
    pub fn compare(&self, other: &Matrix, f: fn((&f64, &f64)) -> bool) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        if self.values.len() != other.values.len() {
            return false;
        }
        self.values.iter().zip(other.values.iter()).all(f)
    }

    /// Zeros all values in the self matrix.
    ///
    pub fn zero(&mut self) {
        self.values.iter_mut().for_each(|v: &mut f64| *v = 0.0);
    }

    /// Convolves the matrix with given filter,
    /// adds necessary padding.
    ///
    pub fn convolve(&mut self, filter: &Matrix) -> Result<(), MatrixError> {
        if self.rows < filter.rows || self.cols < filter.cols {
            return Err(MatrixError::FilterToLarge);
        }

        let mut temp = vec![0.0; self.values.len()];

        let fr_pad: usize = filter.rows / 2;
        let fc_pad: usize = filter.cols / 2;

        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut sum: f64 = 0.0;
                for rf in 0..filter.rows {
                    'filter: for cf in 0..filter.cols {
                        if r + rf < fr_pad
                            || c + cf < fc_pad
                            || r + rf - fr_pad >= self.rows
                            || c + cf - fc_pad >= self.cols
                        {
                            continue 'filter;
                        }
                        let rpi: usize = r + rf - fr_pad;
                        let cpi: usize = c + cf - fc_pad;
                        let svo = self.get_at(rpi, cpi);
                        let fvo = filter.get_at(rf, cf);
                        if let (Ok(sv), Ok(fv)) = (svo, fvo) {
                            sum += sv * fv;
                        } else {
                            return Err(MatrixError::Fatal);
                        }
                    }
                }
                temp[r * self.cols + c] = sum;
            }
        }

        self.values = temp.clone();

        Ok(())
    }

    /// Copies other matrix to self if matrix has the same size.
    ///
    pub fn copy_to_self(&mut self, other: &Matrix) -> Result<(), MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::OutsideRange);
        }
        self.values = other.values.clone();

        Ok(())
    }

    /// Applies dot product of two given matrices to the self matrix.
    ///
    pub fn dot(&mut self, a: &Matrix, b: &Matrix) -> Result<(), MatrixError> {
        if a.cols != b.rows || self.rows != a.rows || self.cols != b.cols {
            return Err(MatrixError::NotMatchingSize);
        }

        for sr in 0..self.rows {
            for sc in 0..self.cols {
                let _ = self.set_at(sr, sc, 0.0);
                for ac in 0..a.cols {
                    let sv = self.get_at(sr, sc);
                    let av = a.get_at(sr, ac);
                    let bv = b.get_at(ac, sc);
                    if let (Ok(mut svv), Ok(avv), Ok(bvv)) = (sv, av, bv) {
                        svv += avv * bvv;
                        let _ = self.set_at(sr, sc, svv);
                    } else {
                        return Err(MatrixError::Fatal);
                    }
                }
            }
        }

        Ok(())
    }

    /// Sums up other matrix to self.
    ///
    pub fn sum(&mut self, other: &Matrix) -> Result<(), MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::NotMatchingSize);
        }

        for r in 0..self.rows {
            for c in 0..self.cols {
                if let (Ok(mut sv), Ok(ov)) = (self.get_at(r, c), other.get_at(r, c)) {
                    sv += ov;
                    let _ = self.set_at(r, c, sv);
                } else {
                    return Err(MatrixError::Fatal);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activators::{LeakyReLU, ReLU, Sigmoid, Tanh};
    use crate::float_math::round;

    #[test]
    fn test_create_matrix_and_getters() {
        let (rows, cols): (usize, usize) = (10, 20);
        let m = Matrix::new(rows, cols);
        assert_eq!(m.get_rows_num(), rows);
        assert_eq!(m.get_cols_num(), cols);
    }

    #[test]
    fn test_randomize_matrix() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        m.values
            .iter()
            .for_each(|v: &f64| assert_eq!(true, *v > 0.0 && *v < 1.0));
    }

    #[test]
    fn test_get_at() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
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
            Ok(_) => panic!("unreachable"),
            Err(err) => assert_eq!(err, MatrixError::OutsideRange),
        }
        let v = m.get_at(10, 21);
        match v {
            Ok(_) => panic!("unreachable"),
            Err(err) => assert_eq!(err, MatrixError::OutsideRange),
        }
    }

    #[test]
    fn test_set_at() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        for r in 0..m.get_rows_num() {
            for c in 0..m.get_cols_num() {
                if let Err(err) = m.set_at(r, c, 0.10) {
                    panic!("{:?}", err);
                }
                let v = m.get_at(r, c);
                match v {
                    Ok(v) => assert_eq!(v, 0.10),
                    Err(err) => panic!("{:?}", err),
                }
            }
        }
    }

    #[test]
    fn test_get_row() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        for r in 0..m.get_rows_num() {
            let mut row: Vec<f64> = Vec::new();
            for c in 0..m.get_cols_num() {
                let v = 0.10 * (r * c) as f64;
                if let Err(err) = m.set_at(r, c, v) {
                    panic!("{:?}", err);
                }
                row.push(v)
            }
            let nm = m.get_row(r);
            if let Ok(nm) = nm {
                row.iter()
                    .zip(nm.values.iter())
                    .for_each(|(x, y): (&f64, &f64)| assert_eq!(x, y));
            }
        }
    }

    #[test]
    fn test_activate_matrix_value_sig() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.activate(&Sigmoid::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_activate_matrix_value_tanh() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.activate(&Tanh::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_activate_matrix_value_relu() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.activate(&ReLU::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(*x, *y);
            });
    }

    #[test]
    fn test_activate_matrix_value_leaky_relu() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.activate(&LeakyReLU::new(1.1));
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(*x, *y);
            });
    }

    #[test]
    fn test_de_activate_matrix_value_sig() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.de_activate(&Sigmoid::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_de_activate_matrix_value_tanh() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.de_activate(&Tanh::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_ne!(*x, *y);
            });
    }

    #[test]
    fn test_de_activate_matrix_value_relu() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.de_activate(&ReLU::new());
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(*x, *y);
            });
    }

    #[test]
    fn test_de_activate_matrix_value_leaky_relu() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        let cp_m = m.clone();
        m.de_activate(&LeakyReLU::new(1.1));
        m.values
            .iter()
            .zip(cp_m.values.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(*x, *y);
            });
    }

    #[test]
    fn test_normalize_unarmalize_values() {
        let (rows, cols): (usize, usize) = (10, 10);
        let mut m = Matrix::new(rows, cols);
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
        };
        let (min, max) = m.min_max();
        assert_eq!(min, 0.0);
        assert_eq!(max, 100.0);
    }

    #[test]
    fn test_compare_success() {
        let m0 = Matrix {
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
        };
        let m1 = Matrix {
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
        };
        assert_eq!(true, m0.compare(&m1, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_compare_failure() {
        let m0 = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0,
                10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 100.0,
                6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0,
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 1.0,
            ],
        };
        let m1 = Matrix {
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
        };
        assert_eq!(false, m0.compare(&m1, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_zero() {
        let (rows, cols): (usize, usize) = (10, 10);
        let mut m = Matrix::new(rows, cols);
        m.randomize();
        m.zero();
        m.values.iter().for_each(|v: &f64| {
            assert_eq!(0.0, *v);
        })
    }

    #[test]
    fn test_convolution_small_matrix() {
        let mut m = Matrix {
            rows: 3,
            cols: 3,
            values: vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        };
        let f = Matrix {
            rows: 3,
            cols: 3,
            values: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };
        let r = Matrix {
            rows: 3,
            cols: 3,
            values: vec![
                6.000, 8.000, 6.000, 9.000, 12.000, 9.000, 6.000, 8.000, 6.000,
            ],
        };

        if let Err(err) = m.convolve(&f) {
            panic!("error: {:?}", err);
        }
        assert_eq!(true, m.compare(&r, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_convolution_large_matrix() {
        let mut m = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0,
            ],
        };
        let f = Matrix {
            rows: 5,
            cols: 5,
            values: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        };
        let r = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                40.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 40.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                60.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 90.000, 60.000,
                40.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 60.000, 40.000,
            ],
        };

        if let Err(err) = m.convolve(&f) {
            panic!("error: {:?}", err);
        }
        assert_eq!(true, m.compare(&r, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_copy_to_self() {
        let (rows, cols): (usize, usize) = (10, 10);
        let mut origin = Matrix::new(rows, cols);
        origin.randomize();
        let mut copy = Matrix::new(rows, cols);
        if let Err(err) = copy.copy_to_self(&origin) {
            panic!("error: {:?}", err);
        }
        assert_eq!(true, origin.compare(&copy, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_dot_0() {
        let a = Matrix {
            rows: 3,
            cols: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let b = Matrix {
            rows: 3,
            cols: 2,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let result = Matrix {
            rows: 3,
            cols: 2,
            values: vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0],
        };
        let mut receiver = Matrix::new(3, 2);

        if let Err(err) = receiver.dot(&a, &b) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            receiver.compare(&result, |(x, y): (&f64, &f64)| *x == *y)
        );
    }

    #[test]
    fn test_dot_1() {
        let a = Matrix {
            rows: 2,
            cols: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let b = Matrix {
            rows: 3,
            cols: 2,
            values: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };
        let result = Matrix {
            rows: 2,
            cols: 2,
            values: vec![58.0, 64.0, 139.0, 154.0],
        };
        let mut receiver = Matrix::new(2, 2);

        if let Err(err) = receiver.dot(&a, &b) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            receiver.compare(&result, |(x, y): (&f64, &f64)| *x == *y)
        );
    }

    #[test]
    fn test_dot_2() {
        let a = Matrix {
            rows: 3,
            cols: 4,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        };
        let b = Matrix {
            rows: 4,
            cols: 2,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let result = Matrix {
            rows: 3,
            cols: 2,
            values: vec![50.0, 60.0, 114.0, 140.0, 178.0, 220.0],
        };
        let mut receiver = Matrix::new(3, 2);

        if let Err(err) = receiver.dot(&a, &b) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            receiver.compare(&result, |(x, y): (&f64, &f64)| *x == *y)
        );
    }

    #[test]
    fn test_sum_0() {
        let mut receiver = Matrix {
            rows: 3,
            cols: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let a = Matrix {
            rows: 3,
            cols: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let result = Matrix {
            rows: 3,
            cols: 3,
            values: vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        };

        if let Err(err) = receiver.sum(&a) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            receiver.compare(&result, |(x, y): (&f64, &f64)| *x == *y)
        );
    }

    #[test]
    fn test_sum_1() {
        let mut receiver = Matrix {
            rows: 3,
            cols: 4,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        };
        let a = Matrix {
            rows: 3,
            cols: 4,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        };
        let result = Matrix {
            rows: 3,
            cols: 4,
            values: vec![
                2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
            ],
        };

        if let Err(err) = receiver.sum(&a) {
            panic!("error: {:?}", err);
        }

        assert_eq!(
            true,
            receiver.compare(&result, |(x, y): (&f64, &f64)| *x == *y)
        );
    }
}
