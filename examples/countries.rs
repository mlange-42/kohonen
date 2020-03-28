use easy_graph::ui::window::WindowBuilder;
use kohonen::calc::neighborhood::Neighborhood;
use kohonen::map::som::DecayParam;
use kohonen::proc::{InputLayer, ProcessorBuilder};
use kohonen::ui::LayerView;

fn main() {
    let layers = vec![
        InputLayer::cont_simple(&[
            "child_mort_2010",
            "birth_p_1000",
            "GNI",
            "LifeExpectancy",
            "PopGrowth",
            "PopUrbanized",
            "PopGrowthUrb",
            "AdultLiteracy",
            "PrimSchool",
            "Income_low_40",
            "Income_high_20",
        ]),
        //InputLayer::cat_simple("species"),
    ];

    let proc = ProcessorBuilder::new(&layers)
        .with_delimiter(b';')
        .with_no_data("-")
        .build_from_file("example_data/countries.csv")
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
        .with_dimensions(1000, 500)
        .with_fps_skip(5.0)
        .build();

    let mut view_x = LayerView::new(win_x, &[0], &proc.data().names_ref_vec(), None);

    while view_x.is_open() {
        som.epoch(proc.data(), None);
        view_x.draw(&som);
    }
}