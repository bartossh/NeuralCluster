/// ActivatiorDeactivatior trait allows to activate and deactivate value
pub trait ActivatiorDeactivatior {
    fn act_f(self, x: f64) -> f64;
    fn deatc_f(self, x: f64) -> f64;
}
