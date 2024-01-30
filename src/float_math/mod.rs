pub fn round(num: f64, places: usize) -> f64 {
    let factor = 10.0_f64.powi(places as i32);
    (num * factor).round() / factor
}
