use easy_graph::color::style::{BLACK, BLUE, WHITE};
use easy_graph::ui::drawing::IntoDrawingArea;
use easy_graph::ui::element::{Circle, PathElement};
use easy_graph::ui::window::WindowBuilder;
use kohonen::calc::neighborhood::Neighborhood;
use kohonen::data::DataFrame;
use kohonen::map::som::{DecayParam, Som, SomParams};
use rand::prelude::*;
use std::time::Instant;

fn main() {
    run_som(true);
}

fn run_som(graphics: bool) {
    let cols = ["A", "B"];
    let params = SomParams::simple(
        1000,
        Neighborhood::Gauss,
        DecayParam::lin(0.1, 0.01),
        DecayParam::lin(10.0, 0.6),
        DecayParam::exp(0.25, 0.0001),
    );
    let mut som = Som::new(cols.len(), 20, 16, params);

    let mut rng = rand::thread_rng();
    let mut data = DataFrame::<f64>::empty(&cols);

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
            win.draw(|b| {
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
