use easy_graph::ui::window::WindowBuilder;
use kohonen::calc::neighborhood::Neighborhood;
use kohonen::map::som::{DecayParam, Layer, Som, SomParams};
use kohonen::ui::LayerView;

fn main() {
    let cols = ["A", "B", "C", "D", "E"];
    let params = SomParams::xyf(
        1000,
        Neighborhood::Gauss,
        DecayParam::lin(0.1, 0.01),
        DecayParam::lin(10.0, 0.6),
        DecayParam::exp(0.25, 0.0001),
        vec![Layer::cont(3, 0.5), Layer::cat(2, 0.5)],
    );
    let som = Som::new(&cols, 16, 20, params);

    let win = WindowBuilder::new()
        .with_dimensions(800, 600)
        .with_fps_skip(5.0)
        .build();

    let mut view = LayerView::new(win, &[0], &cols, None);

    while view.is_open() {
        view.draw(&som, None);
    }
}
