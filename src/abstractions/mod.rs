/// ActivatiorDeactivatior trait allows to activate and deactivate value
///
pub trait ActivatiorDeactivatior {
    fn act_f(&self, x: &mut f64);
    fn de_act_f(&self, x: &mut f64);
}
