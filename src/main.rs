use easy_graph::ui::window::WindowBuilder;
use kohonen::cli::{Cli, CliParsed};
use kohonen::proc::ProcessorBuilder;
use kohonen::ui::LayerView;
use std::time::Duration;
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
        parsed.alpha.clone(),
        parsed.radius.clone(),
        parsed.decay.clone(),
    );

    let mut viewers: Option<Vec<LayerView>> = if parsed.gui {
        Some(
            proc.layers()
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let win = WindowBuilder::new()
                        .with_dimensions(800, 700)
                        .with_fps_skip(parsed.fps)
                        .build();
                    LayerView::new(win, &[i], &proc.data().names_ref_vec(), None)
                })
                .collect(),
        )
    } else {
        None
    };

    if let Some(views) = &mut viewers {
        while views.iter().fold(false, |a, v| a || v.is_open()) {
            let res = som.epoch(&proc.data(), None);
            for view in views.iter_mut() {
                view.draw(&som);
            }
            if res.is_none() {
                std::thread::sleep(Duration::from_millis(40));
            }
        }
    } else {
        while let Some(()) = som.epoch(&proc.data(), None) {}
    }

    if let Some(out) = parsed.output {
        let units_file = out + "-units.csv";
        proc.write_som_units(&som, &units_file, true).unwrap();
    }
}
