use crate::abstractions::ActivatiorDeactivatior;

/// Sigmoid is an entity allowing to perform sigmid activation and deactivation
///
#[derive(Copy, Clone, Debug)]
pub struct Sigmoid;

impl Sigmoid {
    /// new returns new instance of Sigmoid
    ///
    pub fn new() -> Sigmoid {
        Sigmoid {}
    }
}

impl ActivatiorDeactivatior for Sigmoid {
    fn act_f(self, x: &mut f64) {
        *x = 1.0 / (1.0 + f64::exp(-*x));
    }

    fn de_act_f(self, x: &mut f64) {
        *x = *x * (1.0 - *x);
    }
}

/// Tanh is an entity allowing to perform tangens activation and deactivation
///
#[derive(Copy, Clone, Debug)]
pub struct Tanh;

impl Tanh {
    /// new returns new instance of Tanh
    ///
    pub fn new() -> Tanh {
        Tanh {}
    }
}

impl ActivatiorDeactivatior for Tanh {
    fn act_f(self, x: &mut f64) {
        *x = x.tanh();
    }

    fn de_act_f(self, x: &mut f64) {
        *x = x.atanh();
    }
}

/// ReLU is an entity allowing to perform Rectified Linear Unit activation and deactivation
///
#[derive(Copy, Clone, Debug)]
pub struct ReLU;

impl ReLU {
    /// new returns new instance of ReLU
    ///
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl ActivatiorDeactivatior for ReLU {
    fn act_f(self, x: &mut f64) {
        *x = x.max(0.0);
    }

    fn de_act_f(self, x: &mut f64) {
        *x = x.max(0.0);
    }
}

/// LeakyReLU is an entity allowing to perform Leaky Rectified Linear Unit activation and deactivation
///
#[derive(Copy, Clone, Debug)]
pub struct LeakyReLU {
    alpha: f64,
}

impl LeakyReLU {
    /// new returns new instance of LeakyReLU
    ///
    pub fn new(alpha: f64) -> LeakyReLU {
        LeakyReLU { alpha: alpha }
    }
}

impl ActivatiorDeactivatior for LeakyReLU {
    fn act_f(self, x: &mut f64) {
        *x = match *x {
            x if x > 0.0 => x,
            _ => self.alpha * *x,
        };
    }

    fn de_act_f(self, x: &mut f64) {
        self.act_f(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_math::round;

    #[test]
    fn test_sigmoid_activation() {
        let expected = vec![
            0.659450443181008,
            0.6274133159628694,
            0.5112744335829627,
            0.5223695671485495,
            0.683827609808978,
            0.5481250692986543,
            0.7173674403071109,
            0.5751715857364704,
            0.6987744720534808,
        ];
        let sig = Sigmoid::new();
        let mut vc = vec![
            0.66084617, 0.5211358, 0.04510538, 0.08953804, 0.77141825, 0.19309805, 0.9314405,
            0.30298305, 0.8414688,
        ];
        vc.iter_mut().for_each(|x: &mut f64| sig.act_f(x));

        vc.iter()
            .zip(expected.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(round(*x, 5), round(*y, 5));
            })
    }

    #[test]
    fn test_sigmoid_de_activation() {
        let expected = vec![
            0.2241285095963311,
            0.24955327795836,
            0.0430708846950556,
            0.0815209793929584,
            0.17633213356693753,
            0.15581119308619748,
            0.06385909495974999,
            0.2111843214126975,
            0.13339905862655999,
        ];
        let sig = Sigmoid::new();
        let mut vc = vec![
            0.66084617, 0.5211358, 0.04510538, 0.08953804, 0.77141825, 0.19309805, 0.9314405,
            0.30298305, 0.8414688,
        ];
        vc.iter_mut().for_each(|x: &mut f64| sig.de_act_f(x));

        vc.iter()
            .zip(expected.iter())
            .for_each(|(x, y): (&f64, &f64)| {
                assert_eq!(round(*x, 5), round(*y, 5));
            })
    }
}
