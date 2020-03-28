//! Pre- and post-processing of SOM training data, SOM creation.

use crate::calc::neighborhood::Neighborhood;
use crate::calc::norm::{normalize, DeNorm, Norm};
use crate::data::DataFrame;
use crate::map::som::{DecayParam, Layer, Som, SomParams};
use csv::{ReaderBuilder, StringRecord};
use std::collections::HashSet;
use std::error::Error;

/// Layer definition for input tables.
#[derive(Clone, Debug)]
pub struct InputLayer {
    names: Vec<String>,
    indices: Option<Vec<usize>>,
    num_columns: Option<usize>,
    weight: f64,
    is_class: bool,
    norm: Norm,
    scale: f64,
}

impl InputLayer {
    pub fn new(
        names: &[&str],
        weight: f64,
        is_class: bool,
        norm: Norm,
        scale: Option<f64>,
    ) -> Self {
        assert!(names.len() == 1 || !is_class);
        assert!(norm == Norm::None || !is_class);
        InputLayer {
            names: names.iter().map(|x| (&**x).to_string()).collect(),
            indices: None,
            num_columns: None,
            weight,
            is_class,
            norm,
            scale: scale.unwrap_or(1.0),
        }
    }
    pub fn cat(name: &str, weight: f64) -> Self {
        InputLayer {
            names: vec![name.to_string()],
            indices: None,
            num_columns: None,
            weight,
            is_class: true,
            norm: Norm::None,
            scale: 1.0,
        }
    }
    pub fn cat_simple(name: &str) -> Self {
        InputLayer {
            names: vec![name.to_string()],
            indices: None,
            num_columns: None,
            weight: 1.0,
            is_class: true,
            norm: Norm::None,
            scale: 1.0,
        }
    }
    pub fn cont(names: &[&str], weight: f64, norm: Norm, scale: Option<f64>) -> Self {
        InputLayer {
            names: names.iter().map(|x| (&**x).to_string()).collect(),
            indices: None,
            num_columns: None,
            weight,
            is_class: false,
            norm,
            scale: scale.unwrap_or(1.0),
        }
    }
    pub fn cont_simple(names: &[&str]) -> Self {
        InputLayer {
            names: names.iter().map(|x| (&**x).to_string()).collect(),
            indices: None,
            num_columns: None,
            weight: 1.0,
            is_class: false,
            norm: Norm::Gauss,
            scale: 1.0,
        }
    }
}

pub struct ProcessorBuilder {
    input_layers: Vec<InputLayer>,
    delimiter: u8,
}
impl ProcessorBuilder {
    pub fn new(layers: &[InputLayer]) -> Self {
        ProcessorBuilder {
            input_layers: layers.to_vec(),
            delimiter: b',',
        }
    }
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }
    pub fn build_from_file(self, path: &str) -> Result<Processor, Box<dyn Error>> {
        let proc = Processor::new(self.input_layers, path)?;
        Ok(proc)
    }
}

pub struct Processor {
    input_layers: Vec<InputLayer>,
    data: DataFrame<f64>,
    layers: Vec<Layer>,
    norm: Vec<Norm>,
    denorm: Vec<DeNorm>,
    scale: Vec<f64>,
}

impl Processor {
    fn new(input_layers: Vec<InputLayer>, path: &str) -> Result<Self, Box<dyn Error>> {
        Self::read_file(input_layers, path)
    }

    /// The normalized data.
    pub fn data(&self) -> &DataFrame<f64> {
        &self.data
    }
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    pub fn input_layers(&self) -> &[InputLayer] {
        &self.input_layers
    }
    pub fn norm(&self) -> &[Norm] {
        &self.norm
    }
    pub fn denorm(&self) -> &[DeNorm] {
        &self.denorm
    }
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    fn read_file(
        mut input_layers: Vec<InputLayer>,
        path: &str,
    ) -> Result<Processor, Box<dyn Error>> {
        // Read csv
        let mut reader = ReaderBuilder::new()
            .delimiter(b';')
            .from_path(path)
            .unwrap();
        let header: StringRecord = reader.headers().unwrap().clone();
        let header: Vec<_> = header.iter().collect();

        // find column indices for layers
        for lay in input_layers.iter_mut() {
            lay.indices = Some(
                lay.names
                    .iter()
                    .map(|n| header.iter().position(|n2| n2 == n).unwrap())
                    .collect(),
            );
            lay.num_columns = Some(lay.indices.as_ref().unwrap().len());
        }

        // filter out categorical layers
        let categorical: Vec<_> = input_layers
            .iter()
            .enumerate()
            .filter(|(_i, lay)| lay.is_class)
            .collect();

        // find unique levals of categorical layers
        let mut cat_levels: Vec<_> = vec![HashSet::<String>::new(); input_layers.len()];
        let start_pos = reader.position().clone();
        for record in reader.records() {
            let rec = record?;
            for (idx, lay) in categorical.iter() {
                let v = rec.get(lay.indices.as_ref().unwrap()[0]).unwrap();
                let levels = &mut cat_levels[*idx];
                if !levels.contains(v) {
                    levels.insert(v.to_string());
                }
            }
        }

        // convert levels to sorted vectors
        let mut cat_levels: Vec<_> = cat_levels
            .into_iter()
            .map(|levels| {
                let mut lev: Vec<_> = levels.into_iter().collect();
                lev.sort();
                lev
            })
            .collect();

        // determine number of output table columns for categorical layers
        for (cat, levels) in input_layers.iter_mut().zip(cat_levels.iter_mut()) {
            if !levels.is_empty() {
                cat.num_columns = Some(levels.len());
            }
        }

        // create layer definitions
        let weight_scale = 1.0 / input_layers.iter().map(|l| l.weight).sum::<f64>();
        let mut layers = Vec::<Layer>::new();
        let mut colnames = Vec::<String>::new();
        for (idx, lay) in input_layers.iter().enumerate() {
            layers.push(Layer::new(
                lay.num_columns.unwrap(),
                weight_scale * lay.weight,
                lay.is_class,
            ));
            if lay.is_class {
                let base = lay.names[0].clone() + ":";
                let levels = &cat_levels[idx];
                colnames.extend(levels.iter().map(|l| base.clone() + l));
            } else {
                colnames.extend(lay.names.iter().cloned());
            }
        }
        let mut df = DataFrame::<f64>::empty(&colnames.iter().map(|x| &**x).collect::<Vec<_>>());
        let mut row = vec![0.0; colnames.len()];

        reader.seek(start_pos).unwrap();
        for record in reader.records() {
            let rec = record?;
            for i in 0..row.len() {
                row[i] = 0.0;
            }
            let mut start = 0;
            for (layer_index, (inp, lay)) in input_layers.iter().zip(layers.iter()).enumerate() {
                let indices = inp.indices.as_ref().unwrap();
                if inp.is_class {
                    let v = rec.get(indices[0]).unwrap();
                    let pos = cat_levels[layer_index]
                        .iter()
                        .position(|v2| v == v2)
                        .unwrap();
                    row[start + pos] = 1.0;
                } else {
                    for (i, idx) in inp.indices.as_ref().unwrap().iter().enumerate() {
                        let v: f64 = rec.get(*idx).unwrap().parse()?;
                        row[start + i] = v;
                    }
                }
                start += lay.ncols();
            }
            df.push_row(&row);
        }

        let mut norm = Vec::new();
        let mut scale = Vec::new();
        for inp in input_layers.iter() {
            for _ in 0..inp.num_columns.unwrap() {
                norm.push(inp.norm.clone());
                scale.push(inp.scale);
            }
        }
        let (data_norm, denorm) = normalize(&df, &norm, &scale);
        /*
        for row in df.iter_rows() {
            println!("{:?}", row);
        }
        println!("{:?}", cat_levels);
        println!("{:?}", df.names());
        println!("{:?}", norm);
        println!("{:?}", denorm);
        */

        Ok(Processor {
            input_layers,
            data: data_norm,
            layers,
            norm,
            denorm,
            scale,
        })
    }

    pub fn create_som(
        &self,
        nrows: usize,
        ncols: usize,
        epochs: u32,
        neighborhood: Neighborhood,
        alpha: DecayParam,
        radius: DecayParam,
        decay: DecayParam,
    ) -> Som {
        let params = SomParams::xyf(
            epochs,
            neighborhood,
            alpha,
            radius,
            decay,
            self.layers.to_vec(),
        );

        Som::new(self.data.ncols(), nrows, ncols, params)
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::Neighborhood;
    use crate::calc::norm::Norm;
    use crate::map::som::DecayParam;
    use crate::proc::{InputLayer, ProcessorBuilder};

    #[test]
    fn create_proc() {
        let layers = vec![
            InputLayer::cont_simple(&[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]),
            InputLayer::cat_simple("species"),
        ];

        let proc = ProcessorBuilder::new(&layers)
            .with_delimiter(b';')
            .build_from_file("example_data/iris.csv")
            .unwrap();

        let som = proc.create_som(
            16,
            20,
            1000,
            Neighborhood::Gauss,
            DecayParam::lin(0.2, 0.01),
            DecayParam::lin(8.0, 0.5),
            DecayParam::exp(0.2, 0.001),
        );

        assert_eq!(proc.data().nrows(), 150);
        assert_eq!(proc.data().ncols(), 7);
        assert_eq!(proc.data().names().len(), 7);
        assert_eq!(
            proc.norm(),
            &[
                Norm::Gauss,
                Norm::Gauss,
                Norm::Gauss,
                Norm::Gauss,
                Norm::None,
                Norm::None,
                Norm::None
            ]
        );
        assert_eq!(som.weights().ncols(), proc.data().ncols());
        /*
        let win = WindowBuilder::new()
            .with_dimensions(800, 600)
            .with_fps_skip(5.0)
            .build();

        let mut view = LayerView::new(win, &[], None);

        while view.is_open() {
            som.epoch(proc.data(), None);
            view.draw(&som);
        }*/
    }
}
