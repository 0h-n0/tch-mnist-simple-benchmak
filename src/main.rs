use std::time::Instant;

use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Device::Cuda, Tensor};

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 3, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 3, Default::default());
        let fc1 = nn::linear(vs, 9216, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .dropout_(0.25, train)
            .view([-1, 9216])
            .apply(&self.fc1)
            .relu()
            .dropout_(0.5, train)
            .apply(&self.fc2)
    }
}

pub fn run() -> failure::Fallible<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let d = Device::Cuda(0);
    let vs = nn::VarStore::new(d);
    tch::Cuda::cudnn_set_benchmark(true);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    let mut times: Vec<f64> = vec![];
    for epoch in 1..50 {
        let start = Instant::now();
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        let end = start.elapsed();
        times.push(end.as_millis() as f64 / 1000.);
        println!(
            "time {:7.4}s: epoch: {:4} test acc: {:5.2}%",
            end.as_millis() as f64 / 1000.,
            epoch,
            100. * test_accuracy,
        );
    }
    let sum: f64 = times.iter().sum();
    let num: f64 = times.len() as f64;

    println!("{:?} s", sum / num);
    Ok(())
}

fn main() {
    run();
}
