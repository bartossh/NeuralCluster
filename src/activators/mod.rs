use crate::abstractions::ActivatiorDeactivatior;

#[derive(Copy, Clone, Debug)]
pub struct Sigmoid;

impl Sigmoid {
    /// new returns new instance of sigmoid
    pub fn new() -> Sigmoid {
        Sigmoid {}
    }
}

impl ActivatiorDeactivatior for Sigmoid {
    fn act_f(self, x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    fn deatc_f(self, x: f64) -> f64 {
        x * (1.0 - x)
    }
}
