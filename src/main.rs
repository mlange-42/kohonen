use easy_graph::color::style::{BLACK, BLUE, WHITE};
use easy_graph::ui::drawing::bitmap_pixel::RGBPixel;
use easy_graph::ui::drawing::{BitMapBackend, IntoDrawingArea};
use easy_graph::ui::element::{Circle, PathElement};
use easy_graph::ui::window::WindowBuilder;
use kohonen::calc::neighborhood::GaussNeighborhood;
use kohonen::calc::nn;
use kohonen::data::DataFrame;
use kohonen::map::som::{DecayParam, Layer, Som, SomParams};
use kohonen::ui::layer_view::LayerView;
use rand::prelude::*;
use std::time::{Duration, Instant};

fn main() {
    //run_layer_view();
    run_xyf(true);
    //run_som(true);
}

#[allow(dead_code)]
fn run_layer_view() {
    let dim = 5;
    let params = SomParams::xyf(
        1000,
        GaussNeighborhood(),
        DecayParam::lin(0.1, 0.01),
        DecayParam::lin(10.0, 0.6),
        DecayParam::exp(0.25, 0.0001),
        vec![Layer::cont(3, 0.5), Layer::cat(2, 0.5)],
    );
    let som = Som::new(dim, 16, 20, params);

    let win = WindowBuilder::new()
        .with_dimensions(800, 600)
        .with_fps_skip(1.0)
        .build();

    let mut view = LayerView::new(win, &[0], None);

    while view.is_open() {
        view.draw(&som);
    }
}

#[allow(dead_code)]
fn parallel_nn() {
    let mut rng = rand::thread_rng();
    let from = [0.0, 0.0, 0.0];
    let mut to = DataFrame::<f64>::empty(3);

    for _i in 0..1000 {
        to.push_row(&[
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
        ]);
    }

    let (idx, dist) = nn::par_nearest_neighbor(&from, &to, 8);

    println!("{}, {}", idx, dist);
}

#[allow(dead_code)]
fn run_xyf(graphics: bool) {
    let dim = 4;
    let params = SomParams::xyf(
        1000,
        GaussNeighborhood(),
        DecayParam::lin(0.1, 0.01),
        DecayParam::lin(10.0, 0.6),
        DecayParam::exp(0.25, 0.0001),
        vec![Layer::cont(2, 0.5), Layer::cat(2, 0.5)],
    );
    let mut som = Som::new(dim, 12, 24, params);

    let mut rng = rand::thread_rng();
    let mut data = DataFrame::<f64>::empty(dim);

    let norm = rand::distributions::Normal::new(0.0, 0.06);
    for _i in 0..5000 {
        let x: f64 = rng.gen_range(0.1, 0.9);
        let y = 0.25 + 2.0 * (x - 0.5).powi(2) + norm.sample(&mut rng);

        let a = if y > 0.25 { 1.0 } else { 0.1 };
        let b = 1.0 - a;
        data.push_row(&[x, y, a, b]);
    }

    let mut window = if graphics {
        Some(
            WindowBuilder::new()
                .with_dimensions(500, 500)
                .with_fps_skip(2.0)
                .build(),
        )
    } else {
        None
    };

    let mut viewer = if graphics {
        let win = WindowBuilder::new()
            .with_dimensions(800, 500)
            .with_fps_skip(2.0)
            .build();
        Some(LayerView::new(win, &[], None))
    } else {
        None
    };

    let start = Instant::now();

    if let Some(win) = &mut window {
        while win.is_open() || viewer.as_ref().unwrap().is_open() {
            som.epoch(&data, None);

            viewer.as_mut().unwrap().draw(&som);

            win.draw(|b: BitMapBackend<RGBPixel>| {
                let root = b.into_drawing_area();
                root.fill(&WHITE).unwrap();

                for row in data.iter_rows() {
                    let pt1 = ((500.0 * row[0]) as i32, (500.0 * row[1]) as i32);
                    root.draw(&Circle::new(pt1, 1, &BLUE)).unwrap();
                }

                for row in 0..som.nrows() {
                    for col in 0..som.ncols() {
                        let wt1 = som.weights_at(row, col);
                        let pt1 = ((500.0 * wt1[0]) as i32, (500.0 * wt1[1]) as i32);
                        if row < som.nrows() - 1 {
                            let wt2 = som.weights_at(row + 1, col);
                            let pt2 = ((500.0 * wt2[0]) as i32, (500.0 * wt2[1]) as i32);
                            root.draw(&PathElement::new(vec![pt1, pt2], &BLACK))
                                .unwrap();
                        }
                        if col < som.ncols() - 1 {
                            let wt3 = som.weights_at(row, col + 1);
                            let pt3 = ((500.0 * wt3[0]) as i32, (500.0 * wt3[1]) as i32);
                            root.draw(&PathElement::new(vec![pt1, pt3], &BLACK))
                                .unwrap();
                        }
                    }
                }
            });
        }
    } else {
        while let Some(()) = som.epoch(&data, None) {}
    }
    println!("{:?}", start.elapsed());
}

#[allow(dead_code)]
fn run_som(graphics: bool) {
    let dim = 2;
    let params = SomParams::simple(
        1000,
        GaussNeighborhood(),
        DecayParam::lin(0.1, 0.01),
        DecayParam::lin(10.0, 0.6),
        DecayParam::exp(0.25, 0.0001),
    );
    let mut som = Som::new(dim, 20, 16, params);

    let mut rng = rand::thread_rng();
    let mut data = DataFrame::<f64>::empty(dim);

    let norm = rand::distributions::Normal::new(0.0, 0.06);
    for _i in 0..5000 {
        let x: f64 = rng.gen_range(0.1, 0.9);
        let y = 0.25 + 2.0 * (x - 0.5).powi(2) + norm.sample(&mut rng);
        //data.push_row(&[rng.gen_range(0.0, 1.0), rng.gen_range(0.0, 1.0)]);
        data.push_row(&[x, y]);
    }

    let mut window = if graphics {
        Some(
            WindowBuilder::new()
                .with_dimensions(500, 500)
                .with_fps_skip(2.0)
                .build(),
        )
    } else {
        None
    };

    let start = Instant::now();

    if let Some(win) = &mut window {
        while win.is_open() {
            som.epoch(&data, None);
            win.draw(|b: BitMapBackend<RGBPixel>| {
                let root = b.into_drawing_area();
                root.fill(&WHITE).unwrap();

                for row in data.iter_rows() {
                    let pt1 = ((500.0 * row[0]) as i32, (500.0 * row[1]) as i32);
                    root.draw(&Circle::new(pt1, 1, &BLUE)).unwrap();
                }

                for row in 0..som.nrows() {
                    for col in 0..som.ncols() {
                        let wt1 = som.weights_at(row, col);
                        let pt1 = ((500.0 * wt1[0]) as i32, (500.0 * wt1[1]) as i32);
                        if row < som.nrows() - 1 {
                            let wt2 = som.weights_at(row + 1, col);
                            let pt2 = ((500.0 * wt2[0]) as i32, (500.0 * wt2[1]) as i32);
                            root.draw(&PathElement::new(vec![pt1, pt2], &BLACK))
                                .unwrap();
                        }
                        if col < som.ncols() - 1 {
                            let wt3 = som.weights_at(row, col + 1);
                            let pt3 = ((500.0 * wt3[0]) as i32, (500.0 * wt3[1]) as i32);
                            root.draw(&PathElement::new(vec![pt1, pt3], &BLACK))
                                .unwrap();
                        }
                    }
                }
            });
        }
    } else {
        while let Some(()) = som.epoch(&data, None) {}
    }
    println!("{:?}", start.elapsed());
}

#[allow(dead_code)]
fn bench_brute_force(from_count: usize, to_count: usize) -> Duration {
    let from = create_data(from_count);
    let to = create_data(to_count);

    let result = vec![(0, 0.0); from.nrows()];

    let start = Instant::now();
    let _result = nn::nearest_neighbors(&from, &to, result);
    start.elapsed()
}

#[allow(dead_code)]
fn create_data(rows: usize) -> DataFrame<f64> {
    let mut rng = rand::thread_rng();
    let mut df = DataFrame::<f64>::empty(3);

    for _i in 0..rows {
        df.push_row(&[
            rng.gen_range(-1.0, 1.0),
            rng.gen_range(-1.0, 1.0),
            rng.gen_range(-1.0, 1.0),
        ]);
    }
    df
}
