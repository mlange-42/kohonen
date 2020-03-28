use easy_graph::ui::window::WindowBuilder;
use kohonen::cli::{Cli, CliParsed};
use kohonen::proc::ProcessorBuilder;
use kohonen::ui::LayerView;
use structopt::StructOpt;

fn main() {
    let args = Cli::from_args();
    let parsed = CliParsed::from_cli(args);
    println!("{:#?}", parsed);

    let proc = ProcessorBuilder::new(&parsed.layers)
        .with_delimiter(b';')
        .with_no_data(&parsed.no_data)
        .build_from_file(&parsed.file)
        .unwrap();

    let mut som = proc.create_som(
        parsed.size.1,
        parsed.size.0,
        parsed.episodes,
        parsed.neigh.clone(),
        parsed.alpha,
        parsed.radius,
        parsed.decay,
    );

    let mut viewer = if parsed.gui {
        let win = WindowBuilder::new()
            .with_dimensions(1000, 750)
            .with_fps_skip(2.0)
            .build();
        Some(LayerView::new(win, &[], None))
    } else {
        None
    };

    if let Some(view) = &mut viewer {
        while view.is_open() {
            som.epoch(&proc.data(), None);
            view.draw(&som);
        }
    } else {
        while let Some(()) = som.epoch(&proc.data(), None) {}
    }
}
