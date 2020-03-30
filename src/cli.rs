//! Command-line interface for SOMs.
use crate::calc::metric::Metric;
use crate::calc::neighborhood::Neighborhood;
use crate::calc::norm::Norm;
use crate::map::som::{DecayFunction, DecayParam};
use crate::proc::InputLayer;
use crate::EnumFromString;
use structopt::StructOpt;

/// Raw command line arguments.
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
    /// Number of training epochs.
    #[structopt(short, long)]
    epochs: u32,
    /// Layer columns. Put layers in quotes: `"X1 X2 X3" "Y1"`
    #[structopt(short, long)]
    layers: Vec<String>,
    /// Columns to be preserved for output (e.g. item id). Optional, default: none.
    #[structopt(long)]
    preserve: Vec<String>,
    /// Column to be used as label in visualization. Optional, default: none.
    #[structopt(long)]
    labels: Option<String>,
    /// Maximum length of labels. Longer labels are truncated. Optional, default: no limit.
    #[structopt(long = "label-length")]
    label_length: Option<usize>,
    /// Number of labels to show; random sample size. Optional, default: all.
    #[structopt(long = "label-samples")]
    label_samples: Option<usize>,
    /// Layer weights list. Optional, default: '1.0 1.0 ...'
    #[structopt(short, long)]
    weights: Vec<f64>,
    /// Are layers categorical list (true/false). Optional, default: 'false false ...'
    #[structopt(short, long)]
    categ: Vec<bool>,
    /// Distance metric per layer. Optional, default: 'euclidean' for non-categorical, 'tanimoto' for categorical.
    #[structopt(long)]
    metric: Vec<String>,
    /// Normalizer per layer list (gauss, unit, none). Optional, default: 'gauss' for non-categorical, 'none' for categorical.
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
    /// Neighborhood function (gauss|<todo>). Optional, default 'gauss'.
    #[structopt(short = "-g", long)]
    neigh: Option<String>,
    /// Disable GUI
    #[structopt(long = "--no-gui")]
    nogui: bool,
    /// Maximum GUI update frequency in frames per second. Optional, default: '2.0'
    #[structopt(long = "--fps")]
    fps: Option<f64>,
    /// No-data value. Optional, default 'NA'.
    #[structopt(long = "--no-data")]
    no_data: Option<String>,
    /// Output base path, with base file name. Optional, default: no file output.
    #[structopt(short, long)]
    output: Option<String>,
}

/// Parsed command line arguments.
#[derive(Debug)]
pub struct CliParsed {
    pub file: String,
    pub size: (usize, usize),
    pub epochs: u32,
    pub layers: Vec<InputLayer>,
    pub preserve: Vec<String>,
    pub labels: Option<String>,
    pub label_length: Option<usize>,
    pub label_samples: Option<usize>,
    pub alpha: DecayParam,
    pub radius: DecayParam,
    pub decay: DecayParam,
    pub neigh: Neighborhood,
    pub gui: bool,
    pub no_data: String,
    pub fps: f64,
    pub output: Option<String>,
}

impl CliParsed {
    /// Parse arguments from a [`Cli`](struct.Cli.html).
    pub fn from_cli(mut cli: Cli) -> Self {
        CliParsed {
            file: cli.file.clone(),
            size: (cli.size[0], cli.size[1]),
            epochs: cli.epochs,
            layers: Self::parse_layers(&mut cli),
            preserve: cli.preserve,
            labels: cli.labels,
            label_length: cli.label_length,
            label_samples: cli.label_samples,
            alpha: Self::parse_decay(cli.alpha, "alpha"),
            radius: Self::parse_decay(cli.radius, "radius"),
            decay: Self::parse_decay(cli.decay, "decay"),
            neigh: match &cli.neigh {
                Some(n) => Neighborhood::from_string(n).unwrap(),
                None => Neighborhood::Gauss,
            },
            gui: !cli.nogui,
            no_data: cli.no_data.unwrap_or("NA".to_string()),
            fps: cli.fps.unwrap_or(2.0),
            output: cli.output,
        }
    }

    fn parse_decay(values: Vec<String>, name: &str) -> DecayParam {
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
    fn parse_layers(cli: &mut Cli) -> Vec<InputLayer> {
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
        if cli.metric.len() != 0 && cli.metric.len() != n_layers {
            panic!("Expected no metric, or as many as layers (option --metric)");
        }
        if cli.norm.len() != 0 && cli.norm.len() != n_layers {
            panic!("Expected no normalizers, or as many as layers (option --norm)");
        }

        if cli.weights.is_empty() {
            cli.weights = vec![1.0; n_layers];
        }
        if cli.categ.is_empty() {
            cli.categ = vec![false; n_layers];
        }
        if cli.norm.is_empty() {
            cli.norm = cli
                .categ
                .iter()
                .map(|c| {
                    if *c {
                        "none".to_string()
                    } else {
                        "gauss".to_string()
                    }
                })
                .collect();
        }
        if cli.metric.is_empty() {
            cli.metric = cli
                .categ
                .iter()
                .map(|c| {
                    if *c {
                        "tanimoto".to_string()
                    } else {
                        "euclidean".to_string()
                    }
                })
                .collect();
        }

        cli.layers
            .iter()
            .zip(&cli.weights)
            .zip(&cli.categ)
            .zip(&cli.metric)
            .zip(&cli.norm)
            .map(|((((lay, wt), cat), metr), norm)| {
                InputLayer::new(
                    &lay.trim().split(' ').map(|s| &*s).collect::<Vec<_>>(),
                    *wt,
                    *cat,
                    Metric::from_string(metr).unwrap(),
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
