use easy_graph::ui::window::WindowBuilder;
use kohonen::calc::neighborhood::Neighborhood;
use kohonen::map::som::DecayParam;
use kohonen::proc::{InputLayer, ProcessorBuilder};
use kohonen::ui::LayerView;

fn main() {
    let layers = vec![
        InputLayer::cont_simple(&["sepal_length", "sepal_width", "petal_length", "petal_width"]),
        InputLayer::cat_simple("species"),
    ];

    let proc = ProcessorBuilder::new(&layers)
        .with_delimiter(b';')
        .build_from_file("example_data/iris.csv")
        .unwrap();

    let mut som = proc.create_som(
        16,
        20,
        5000,
        Neighborhood::Gauss,
        DecayParam::lin(0.2, 0.01),
        DecayParam::lin(8.0, 0.5),
        DecayParam::exp(0.2, 0.001),
    );

    let win_x = WindowBuilder::new()
        .with_position((10, 10))
        .with_dimensions(600, 500)
        .with_fps_skip(5.0)
        .build();

    let win_y = WindowBuilder::new()
        .with_position((620, 10))
        .with_dimensions(600, 500)
        .with_fps_skip(5.0)
        .build();

    let mut view_x = LayerView::new(win_x, &[0], None);
    let mut view_y = LayerView::new(win_y, &[1], None);

    while view_x.is_open() || view_y.is_open() {
        som.epoch(proc.data(), None);
        view_x.draw(&som);
        view_y.draw(&som);
    }
}
