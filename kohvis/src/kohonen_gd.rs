use gdnative::{user_data, Node};
use kohonen::cli::{Cli, CliParsed};
use kohonen::map::som::Som;
use kohonen::proc::{Processor, ProcessorBuilder};
use std::time::Instant;
use std::{env, fs};
use structopt::StructOpt;

#[derive(gdnative::NativeClass)]
#[inherit(gdnative::Node)]
#[user_data(user_data::LocalCellData<Kohonen>)]
pub struct Kohonen {
    processor: Option<Processor>,
    som: Option<Som>,
    cli: Option<CliParsed>,
    done: bool,
}

#[gdnative::methods]
impl Kohonen {
    pub fn processor(&self) -> &Option<Processor> {
        &self.processor
    }
    pub fn som(&self) -> &Option<Som> {
        &self.som
    }
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn _init(_owner: gdnative::Node) -> Self {
        Kohonen {
            cli: None,
            processor: None,
            som: None,
            done: false,
        }
    }

    #[export]
    fn _ready(&mut self, _owner: gdnative::Node) {
        let args: Vec<_> = env::args().collect();

        let parsed: CliParsed = if args.len() == 2 && !args[1].starts_with('-') {
            let mut content = fs::read_to_string(&args[1]).unwrap_or_else(|err| {
                panic!(
                    "Something went wrong reading the options file {:?}: {}",
                    &args[1], err,
                )
            });
            content = "kohonen ".to_string() + &content.replace("\r\n", " ").replace("\n", " ");
            let cli: Cli = content.parse().unwrap();
            CliParsed::from_cli(cli)
        } else {
            let cli = Cli::from_args();
            CliParsed::from_cli(cli)
        };

        godot_print!("{:#?}", parsed);

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

        self.som = Some(proc.create_som(
            parsed.size.1,
            parsed.size.0,
            parsed.epochs,
            parsed.neigh.clone(),
            parsed.alpha.clone(),
            parsed.radius.clone(),
            parsed.decay.clone(),
        ));
        self.processor = Some(proc);
        self.cli = Some(parsed);
    }

    #[export]
    pub fn _process(&mut self, _owner: Node, _delta: f64) {
        if let Some(proc) = &self.processor {
            if let Some(som) = &mut self.som {
                if let Some(cli) = &self.cli {
                    let start = Instant::now();
                    loop {
                        let res = som.epoch(&proc.data(), None);
                        if res.is_none() {
                            if !self.done {
                                println!("Done.");
                                kohonen::write_output(&cli, &proc, &som);
                                self.done = true;
                            }
                            break;
                        }
                        if start.elapsed().as_millis() > 25 {
                            break;
                        }
                    }
                    // godot_print!("{:?}", som.get_epoch());
                }
            }
        }
    }
}
