use crate::car::Car;
use crate::map::OccGrid;

pub fn render_rgb(map: &OccGrid, cars: &[Car]) -> (Vec<u8>, u32, u32) {
    let (w, h) = (map.img.width(), map.img.height());
    let mut buf = vec![0u8; (h * w * 3) as usize];
    for (i, p) in map.img.pixels().enumerate() {
        buf[i * 3..][..3].fill(p.0[0]);
    }
    for car in cars {
        for (x, y) in map.car_pixels(car) {
            if x < w && y < h {
                let i = (y as usize * w as usize + x as usize) * 3;
                buf[i..i + 3].copy_from_slice(&[43, 127, 255]);
            }
        }
    }
    (buf, h, w)
}
