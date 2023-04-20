pub struct MultilayerShape {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub range: usize,
}

impl MultilayerShape {
    pub fn new(x: usize, y: usize, z: usize) -> MultilayerShape {
        MultilayerShape {
            x,
            y,
            z,
            range: x * y * z,
        }
    }
}