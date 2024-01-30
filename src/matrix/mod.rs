use crate::abstractions::ActivatiorDeactivatior;
use rand::rngs::OsRng;
use rand::RngCore;

#[derive(Clone, Debug, PartialEq)]
pub enum MatrixError {
    OutsideRange,
    FilterToLarge,
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

    /// set_at sets value at given row and column
    ///
    pub fn set_at(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutsideRange);
        }
        self.values[row * self.cols + col] = value;

        Ok(())
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

    /// zero zeros all values in matrix
    ///
    pub fn zero(&mut self) {
        self.values.iter_mut().for_each(|v: &mut f64| *v = 0.0);
    }

    /// convolve convolves the matrix with given filter
    ///
    pub fn convolve(&mut self, filter: &Matrix<T>) -> Result<(), MatrixError> {
        if self.rows > filter.rows || self.cols > filter.cols {
            return Err(MatrixError::FilterToLarge);
        }

        let filter_rows_padding: usize = filter.rows / 2;
        let filter_cols_padding: usize = filter.cols / 2;

        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut sum: f64 = 0.0;
                for rp in 0..filter.rows {
                    for cp in 0..filter.cols {
                        let rpi: usize = r + rp - filter_rows_padding;
                        let cpi: usize = c + cp - filter_cols_padding;
                        let svo = self.get_at(rpi, cpi);
                        let fvo = filter.get_at(rp, cp);
                        if let (Ok(sv), Ok(fv)) = (svo, fvo) {
                            sum += sv * fv;
                        }
                    }
                }
                if let Err(err) = self.set_at(r, c, sum) {
                    return Err(err);
                }
            }
        }

        Ok(())
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
    fn test_set_at() {
        let (rows, cols): (usize, usize) = (10, 20);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
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
            act_deact: Sigmoid::new(),
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
            act_deact: Sigmoid::new(),
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
            act_deact: Sigmoid::new(),
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
            act_deact: Sigmoid::new(),
        };
        assert_eq!(false, m0.compare(&m1, |(x, y): (&f64, &f64)| *x == *y));
    }

    #[test]
    fn test_zero() {
        let (rows, cols): (usize, usize) = (10, 10);
        let mut m = Matrix::new(rows, cols, Sigmoid::new());
        m.randomize();
        m.zero();
        m.values.iter().for_each(|v: &f64| {
            assert_eq!(0.0, *v);
        })
    }

    #[test]
    fn test_convolution() {
        let mut m = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
                44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
                58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
                72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
                86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
                100.0,
            ],
            act_deact: Sigmoid::new(),
        };
        let mut f = Matrix {
            rows: 3,
            cols: 3,
            values: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            act_deact: Sigmoid::new(),
        };
        let mut r = Matrix {
            rows: 10,
            cols: 10,
            values: vec![
                1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000, 11.000,
                12.000, 13.000, 14.000, 15.000, 16.000, 17.000, 18.000, 19.000, 20.000, 21.000,
                22.000, 23.000, 24.000, 25.000, 26.000, 27.000, 28.000, 29.000, 30.000, 31.000,
                32.000, 33.000, 34.000, 35.000, 36.000, 37.000, 38.000, 39.000, 40.000, 41.000,
                42.000, 43.000, 44.000, 45.000, 46.000, 47.000, 48.000, 49.000, 50.000, 51.000,
                52.000, 53.000, 54.000, 55.000, 56.000, 57.000, 58.000, 59.000, 60.000, 61.000,
                62.000, 63.000, 64.000, 65.000, 66.000, 67.000, 68.000, 69.000, 70.000, 71.000,
                72.000, 73.000, 74.000, 75.000, 76.000, 77.000, 78.000, 79.000, 80.000, 81.000,
                82.000, 83.000, 84.000, 85.000, 86.000, 87.000, 88.000, 89.000, 90.000, 91.000,
                92.000, 93.000, 94.000, 95.000, 96.000, 97.000, 98.000, 99.000, 100.000,
            ],
            act_deact: Sigmoid::new(),
        };

        m.convolve(&f);
        assert_eq!(true, m.compare(&r, |(x, y): (&f64, &f64)| *x == *y));
    }
}
