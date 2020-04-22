use easy_graph::ui::window::WindowBuilder;
use kohonen::cli::{Cli, CliParsed};
use kohonen::map::som::Som;
use kohonen::proc::{Processor, ProcessorBuilder};
use kohonen::ui::LayerView;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{env, fs};
use structopt::StructOpt;

fn main() {
    let is_test = false;

    let args: Vec<String> = if is_test {
        vec![
            "kohonen".to_string(),
            "cmd_examples/countries-test.koo".to_string(),
        ]
    } else {
        env::args().collect()
    };
    let mut parsed: CliParsed = if args.len() == 2 && !args[1].starts_with('-') {
        let mut content = fs::read_to_string(&args[1]).expect(&format!(
            "Something went wrong reading the options file {:?}",
            &args[1]
        ));
        content = "kohonen ".to_string() + &content.replace("\r\n", " ").replace("\n", " ");
        let cli: Cli = content.parse().unwrap();
        CliParsed::from_cli(cli)
    } else {
        let cli = Cli::from_args();
        CliParsed::from_cli(cli)
    };

    println!("{:#?}", parsed);

    let proc = ProcessorBuilder::new(
        &parsed.layers,
        &parsed.preserve,
        &parsed.labels,
        &parsed.label_length,
        &parsed.label_samples,
    )
    .with_delimiter(b';')
    .with_no_data(&parsed.no_data)
    .build_from_file(&parsed.file)
    .unwrap();

    let mut som = proc.create_som(
        parsed.size.1,
        parsed.size.0,
        parsed.epochs,
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
                        .with_title(&format!("Layer {}", i))
                        .with_dimensions(800, 700)
                        .with_fps_skip(parsed.fps)
                        .build();
                    LayerView::new(win, &[i], &proc.data().columns_ref_vec(), None)
                })
                .collect(),
        )
    } else {
        None
    };

    let mut done = false;

    let start = Instant::now();

    if let Some(views) = &mut viewers {
        while views.iter().fold(false, |a, v| a || v.is_open()) {
            let res = som.epoch(&proc.data(), None);
            let label_data = match proc.labels() {
                Some(lab) => Some((proc.data(), lab)),
                None => None,
            };
            for view in views.iter_mut() {
                view.draw(&som, label_data);
            }
            if res.is_none() {
                if !done {
                    println!("Elapsed: {:?}", start.elapsed());
                    write_output(&parsed, &proc, &som);
                    done = true;
                }
                if parsed.wait {
                    std::thread::sleep(Duration::from_millis(40));
                } else {
                    break;
                }
            }
        }
        parsed.wait = false;
    } else {
        while let Some(()) = som.epoch(&proc.data(), None) {}
        println!("Elapsed: {:?}", start.elapsed());
        write_output(&parsed, &proc, &som);
    }

    if parsed.wait {
        dont_disappear::any_key_to_continue::default();
    }
}

fn write_output(parsed: &CliParsed, proc: &Processor, som: &Som) {
    if let Some(out) = &parsed.output {
        let units_file = format!("{}-units.csv", &out);
        proc.write_som_units(&som, &units_file, true).unwrap();
        let data_file = format!("{}-out.csv", &out);
        proc.write_data_nearest(&som, proc.data(), &data_file)
            .unwrap();
        let norm_file = format!("{}-norm.csv", &out);
        proc.write_normalization(&som, &norm_file).unwrap();

        let som_file = format!("{}-som.json", &out);
        let serialized = serde_json::to_string_pretty(&(som, proc.denorm())).unwrap();
        let mut file = File::create(som_file).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }
}

/*
#[derive(Serialize, Deserialize)]
struct SomSerialization<'a> {
    som: &'a Som,
    denorm: &'a [LinearTransform],
}
*/
