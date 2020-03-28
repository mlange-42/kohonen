//! Command-line interface for SOMs.
use crate::calc::neighborhood::Neighborhood;
use crate::calc::norm::Norm;
use crate::map::som::{DecayFunction, DecayParam};
use crate::proc::InputLayer;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "Super-SOM command line application")]
pub struct Cli {
    // TODO: add and implement no-data value (use countries example)
    /// Path to the training data file.
    #[structopt(short, long)]
    file: String,
    /// SOM size: width, height.
    #[structopt(short, long, number_of_values = 2)]
    size: Vec<usize>,
    /// Number of training episodes
    #[structopt(short, long)]
    episodes: u32,
    /// Layer columns. Put layers in quotes: `"X1 X2 X3" "Y1"`
    #[structopt(short, long)]
    layers: Vec<String>,
    /// Layer weights list
    #[structopt(short, long)]
    weights: Vec<f64>,
    /// Are layers categorical list (0/1). Default 1.0
    #[structopt(short, long)]
    categ: Vec<i32>,
    /// Normalizer per layer list (gauss, unit, none). Default gauss.
    #[structopt(short, long)]
    norm: Vec<String>,
    /// Learning rate: start, end, type (lin|exp)
    #[structopt(short, long, number_of_values = 3)]
    alpha: Vec<String>,
    /// Neighborhood radius: start, end, type (lin|exp)
    #[structopt(short, long, number_of_values = 3)]
    radius: Vec<String>,
    /// Weight decay: start, end, type (lin|exp)
    #[structopt(short, long, number_of_values = 3)]
    decay: Vec<String>,
    /// Neighborhood function (gauss|<todo>)
    #[structopt(short = "-g", long)]
    neigh: Option<String>,
    /// Disable GUI
    #[structopt(long = "--no-gui")]
    nogui: bool,
    /// Disable GUI
    #[structopt(long = "--fps")]
    fps: Option<f64>,
    /// No-data value. Default 'NA'.
    #[structopt(long = "--no-data")]
    no_data: Option<String>,
}

#[derive(Debug)]
pub struct CliParsed {
    pub file: String,
    pub size: (usize, usize),
    pub episodes: u32,
    pub layers: Vec<InputLayer>,
    pub alpha: DecayParam,
    pub radius: DecayParam,
    pub decay: DecayParam,
    pub neigh: Neighborhood,
    pub gui: bool,
    pub no_data: String,
    pub fps: f64,
}

impl CliParsed {
    pub fn from_cli(mut cli: Cli) -> Self {
        CliParsed {
            file: cli.file.clone(),
            size: (cli.size[0], cli.size[1]),
            episodes: cli.episodes,
            layers: Self::to_layers(&mut cli),
            alpha: Self::to_decay(cli.alpha, "alpha"),
            radius: Self::to_decay(cli.radius, "radius"),
            decay: Self::to_decay(cli.decay, "decay"),
            neigh: match &cli.neigh {
                Some(n) => Neighborhood::from_string(n).unwrap(),
                None => Neighborhood::Gauss,
            },
            gui: !cli.nogui,
            no_data: cli.no_data.unwrap_or("NA".to_string()),
            fps: cli.fps.unwrap_or(2.0),
        }
    }

    fn to_decay(values: Vec<String>, name: &str) -> DecayParam {
        if values.len() != 3 {
            panic!(format!(
                "Three argument required for {}: start value, end value, decay function (lin|exp)",
                name
            ));
        }
        DecayParam::new(
            values[0]
                .parse()
                .expect(&format!("Unable to parse value {} in {}", values[0], name)),
            values[1]
                .parse()
                .expect(&format!("Unable to parse value {} in {}", values[1], name)),
            DecayFunction::from_string(&values[2]).unwrap(),
            /*
            match &values[2][..] {
                "lin" => DecayFunction::Linear,
                "exp" => DecayFunction::Exponential,
                _ => panic!("Expected decay funtion 'lin' or 'exp'"),
            },*/
        )
    }
    fn to_layers(cli: &mut Cli) -> Vec<InputLayer> {
        if cli.layers.is_empty() {
            panic!("Expected columns for at least one layer (option --layers)");
        }
        let n_layers = cli.layers.len();

        if cli.weights.len() != 0 && cli.weights.len() != n_layers {
            panic!("Expected no weights, or as many as layers (option --weights)");
        }
        if cli.categ.len() != 0 && cli.categ.len() != n_layers {
            panic!("Expected no categorical 0/1, or as many as layers (option --weights)");
        }
        if cli.norm.len() != 0 && cli.norm.len() != n_layers {
            panic!("Expected no normalizers, or as many as layers (option --norm)");
        }

        if cli.weights.is_empty() {
            cli.weights = vec![1.0; n_layers];
        }
        if cli.categ.is_empty() {
            cli.categ = vec![0; n_layers];
        }
        if cli.norm.is_empty() {
            cli.norm = vec!["gauss".to_string(); n_layers];
        }

        cli.layers
            .iter()
            .zip(&cli.weights)
            .zip(&cli.categ)
            .zip(&cli.norm)
            .map(|(((lay, wt), cat), norm)| {
                InputLayer::new(
                    &lay.trim().split(' ').map(|s| &*s).collect::<Vec<_>>(),
                    *wt,
                    *cat > 0,
                    Norm::from_string(norm).unwrap(),
                    None,
                )
            })
            .collect::<Vec<_>>()
    }
}

/*
mod parse {
    use crate::map::som::DecayFunction;
    use std::convert::TryFrom;

    pub fn parse_decay_param(src: &str) -> (f64, f64, DecayFunction) {
        let split: Vec<_> = src.split(" ").collect();
        (
            split[0].parse().unwrap(),
            split[1].parse().unwrap(),
            match split[2] {
                "lin" => DecayFunction::Linear,
                "exp" => DecayFunction::Exponential,
                _ => panic!("Expected decay funtion 'lin' or 'exp'"),
            },
        )
    }
}
*/
