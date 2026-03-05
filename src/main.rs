mod car;
mod map;
#[cfg(feature = "ros")]
mod ros;
mod sim;
mod skeleton;

#[cfg(not(feature = "ros"))]
use crate::sim::Sim;

#[cfg(not(feature = "ros"))]
fn main() {
    let mut sim = Sim::new("maps/skirk.yaml", 1, 10_000);
    sim.reset();
    for _ in 0..1_000_000 {
        let _obs = sim.step(&[0.0, 0.5]);
    }
}

#[cfg(feature = "ros")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bridge = ros::RosBridge {
        sim: sim::Sim::new("maps/berlin.yaml", 1, 10_000),
        hz: 100.0,
    };
    bridge.spin(vec![[0.0, 0.0, 0.0]]).await
}
