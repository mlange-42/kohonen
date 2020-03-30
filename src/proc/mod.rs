//! Pre- and post-processing of SOM training data, SOM creation.

use crate::calc::neighborhood::Neighborhood;
use crate::calc::nn::nearest_neighbor_xyf;
use crate::calc::norm::{denormalize_columns, normalize, LinearTransform, Norm};
use crate::data::DataFrame;
use crate::map::som::{DecayParam, Layer, Som, SomParams};
use crate::DataTypeError;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
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
    /// Creates a new input layer definition.
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

    /// Creates a new categorical input layer definition.
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

    /// Creates a new categorical input layer definition with default weight.
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

    /// Creates a new continuous / non-categorical input layer definition.
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

    /// Creates a new continuous / non-categorical input layer definition with default weight and Gaussian normalization.
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

/// Csv file options
#[derive(Clone, Debug)]
pub struct CsvOptions {
    delimiter: u8,
    no_data: String,
}

/// Builder for ['Processor'](struct.Processor.html).
pub struct ProcessorBuilder {
    input_layers: Vec<InputLayer>,
    preserve: Vec<String>,
    labels: Option<String>,
    label_length: Option<usize>,
    csv_options: CsvOptions,
}
impl ProcessorBuilder {
    /// Creates a `ProcessorBuilder` for the given [`InputLayer`s](struct.InputLayer.html).
    pub fn new(
        layers: &[InputLayer],
        preserve: &Vec<String>,
        label: &Option<String>,
        label_length: &Option<usize>,
    ) -> Self {
        ProcessorBuilder {
            input_layers: layers.to_vec(),
            preserve: preserve.clone(),
            labels: label.clone(),
            label_length: label_length.clone(),
            csv_options: CsvOptions {
                delimiter: b',',
                no_data: "NA".to_string(),
            },
        }
    }
    /// Sets the delimiter for CSV files. Default ','.
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.csv_options.delimiter = delimiter;
        self
    }
    /// Sets the no-data value for CSV files. Default 'NA'.
    pub fn with_no_data(mut self, no_data: &str) -> Self {
        self.csv_options.no_data = no_data.to_string();
        self
    }
    /// Builds a [`Processor`](struct.Processor.html) from the given data file.
    pub fn build_from_file(self, path: &str) -> Result<Processor, Box<dyn Error>> {
        let proc = Processor::new(
            self.input_layers,
            self.preserve,
            self.labels,
            self.label_length,
            path,
            &self.csv_options,
        )?;
        Ok(proc)
    }
}

/// Central type for SOM setup and processing.
#[allow(dead_code)]
pub struct Processor {
    input_layers: Vec<InputLayer>,
    data: DataFrame,
    layers: Vec<Layer>,
    preserve_columns: Vec<String>,
    preserved: Vec<Vec<String>>,
    label_column: Option<String>,
    labels: Option<Vec<String>>,
    norm: Vec<Norm>,
    denorm: Vec<LinearTransform>,
    scale: Vec<f64>,
    csv_options: CsvOptions,
}

impl Processor {
    fn new(
        input_layers: Vec<InputLayer>,
        preserve: Vec<String>,
        labels: Option<String>,
        label_length: Option<usize>,
        path: &str,
        csv_options: &CsvOptions,
    ) -> Result<Self, Box<dyn Error>> {
        Self::read_file(
            input_layers,
            preserve,
            labels,
            label_length,
            path,
            csv_options,
        )
    }

    /// Return a reference to the normalized data.
    pub fn data(&self) -> &DataFrame {
        &self.data
    }
    /// Return a reference to SOM layer definitions.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    /// Return a reference to input layer definitions.
    pub fn input_layers(&self) -> &[InputLayer] {
        &self.input_layers
    }
    /// Return a reference to the applied normalizers.
    pub fn norm(&self) -> &[Norm] {
        &self.norm
    }
    /// Return a reference to the transforms for de-normalization.
    pub fn denorm(&self) -> &[LinearTransform] {
        &self.denorm
    }
    /// Return a reference the applies scalings (not used so far).
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    pub fn labels(&self) -> Option<&[String]> {
        match &self.labels {
            Some(lab) => Some(&lab),
            None => None,
        }
    }

    fn read_file(
        mut input_layers: Vec<InputLayer>,
        preserve_columns: Vec<String>,
        label_column: Option<String>,
        label_length: Option<usize>,
        path: &str,
        csv_options: &CsvOptions,
    ) -> Result<Processor, Box<dyn Error>> {
        let no_data = &csv_options.no_data;

        // Read csv
        let mut reader = ReaderBuilder::new()
            .delimiter(csv_options.delimiter)
            .from_path(path)
            .unwrap();
        let header: StringRecord = reader.headers().unwrap().clone();
        let header: Vec<_> = header.iter().collect();

        // find column indices for layers
        for lay in input_layers.iter_mut() {
            lay.indices = Some(
                lay.names
                    .iter()
                    .map(|n| {
                        header
                            .iter()
                            .position(|n2| n2 == n)
                            .expect(&format!("Volumn '{}' not found.", n))
                    })
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
                if v != no_data && !levels.contains(v) {
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

        // get id column index
        let id_indices: Vec<_> = preserve_columns
            .iter()
            .map(|col| {
                header
                    .iter()
                    .position(|n2| *n2 == col)
                    .expect(&format!("Preserved column '{}' not found.", col))
            })
            .collect();
        let mut id_values = vec![Vec::<String>::new(); id_indices.len()];

        let (label_index, mut labels) = match &label_column {
            Some(col) => (
                Some(
                    header
                        .iter()
                        .position(|n2| *n2 == col)
                        .expect(&format!("Label column '{}' not found.", col)),
                ),
                Some(Vec::new()),
            ),
            None => (None, None),
        };

        // transform to SOM training data format
        let mut df = DataFrame::empty(&colnames.iter().map(|x| &**x).collect::<Vec<_>>());
        let mut row = vec![0.0; colnames.len()];

        reader.seek(start_pos).unwrap();
        for record in reader.records() {
            let rec = record?;
            for i in 0..row.len() {
                row[i] = 0.0;
            }
            for (idx, col_idx) in id_indices.iter().enumerate() {
                let id = rec.get(*col_idx).unwrap();
                id_values[idx].push(id.to_string());
            }
            if let Some(col_idx) = &label_index {
                let mut id = rec.get(*col_idx).unwrap();
                if let Some(len) = label_length {
                    if id.len() > len {
                        id = &id[..len];
                    }
                }
                labels.as_mut().unwrap().push(id.to_string());
            }
            let mut start = 0;
            for (layer_index, (inp, lay)) in input_layers.iter().zip(layers.iter()).enumerate() {
                let indices = inp.indices.as_ref().unwrap();
                if inp.is_class {
                    let v = rec.get(indices[0]).unwrap();
                    if v == no_data {
                        for i in start..(start + cat_levels[layer_index].len()) {
                            row[i] = std::f64::NAN;
                        }
                    } else {
                        let pos = cat_levels[layer_index]
                            .iter()
                            .position(|v2| v == v2)
                            .unwrap();
                        row[start + pos] = 1.0;
                    }
                } else {
                    for (i, idx) in inp.indices.as_ref().unwrap().iter().enumerate() {
                        let str = rec.get(*idx).unwrap();
                        if str == no_data {
                            row[start + i] = std::f64::NAN;
                        } else {
                            let v: f64 = str.parse().expect(&format!(
                                "Unable to parse value {} in column {}",
                                str, inp.names[i]
                            ));
                            row[start + i] = v;
                        }
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

        Ok(Processor {
            input_layers,
            data: data_norm,
            preserve_columns: preserve_columns.clone(),
            preserved: id_values,
            label_column: label_column.clone(),
            labels,
            layers,
            norm,
            denorm,
            scale,
            csv_options: csv_options.clone(),
        })
    }

    /// Creates an SOM for the `Processor`'s layer definitions and data.
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

        Som::new(&self.data.names_ref_vec(), nrows, ncols, params)
    }

    /// Transforms a categorical / class layer to a vector of class labels.
    ///
    /// Returns an error if the layer is not categorical.
    pub fn to_class(
        &self,
        som: &Som,
        data: &DataFrame,
        layer_index: usize,
    ) -> Result<(String, Vec<String>), DataTypeError> {
        if !self.input_layers[layer_index].is_class {
            return Err(DataTypeError(format!(
                "Classes can be derived only for categorical layers, but layer {} is not.",
                layer_index
            )));
        }
        let layer = &self.layers[layer_index];
        let start_col = som.params().start_columns()[layer_index];

        let classes: Vec<_> = som.weights().names()[start_col..(start_col + layer.ncols())]
            .iter()
            .map(|n| n.splitn(2, ':').nth(1).unwrap())
            .collect();
        let name = self.data.names()[start_col].splitn(2, ':').nth(0).unwrap();

        let no_data = &self.csv_options.no_data;
        let result: Vec<_> = data
            .iter_rows()
            .map(|row| {
                let mut v_max = std::f64::MIN;
                let mut idx_max = 0;
                let mut any = false;
                for i in start_col..(start_col + layer.ncols()) {
                    let v = row[i];
                    if !v.is_nan() {
                        if v > v_max {
                            v_max = v;
                            idx_max = i;
                        }
                        any = true;
                    }
                }
                if any {
                    classes[idx_max - start_col].to_string()
                } else {
                    no_data.clone()
                }
            })
            .collect();

        Ok((name.to_string(), result))
    }

    /// De-normalizes a SOM layer.
    pub fn to_denormalized(
        &self,
        som: &Som,
        data: &DataFrame,
        layer_index: usize,
    ) -> Result<DataFrame, DataTypeError> {
        let layer = &self.layers[layer_index];
        let start_col = som.params().start_columns()[layer_index];
        let range = start_col..(start_col + layer.ncols());
        Ok(denormalize_columns(
            data,
            &range.collect::<Vec<_>>(),
            &self.denorm()[start_col..(start_col + layer.ncols())],
        ))
    }

    /// Writes SOM units to CSV file.
    pub fn write_som_units(
        &self,
        som: &Som,
        path: &str,
        class_values: bool,
    ) -> Result<(), Box<dyn Error>> {
        let mut classes: Vec<Option<Vec<String>>> = vec![None; self.layers.len()];
        let mut denorm: Vec<Option<DataFrame>> = (0..self.layers.len()).map(|_| None).collect();

        let mut names: Vec<String> =
            vec!["index".to_string(), "row".to_string(), "col".to_string()];
        let offset = names.len();
        for (idx, layer) in som.params().layers().iter().enumerate() {
            if class_values || !layer.categorical() {
                let result = self.to_denormalized(&som, som.weights(), idx).unwrap();
                names.extend_from_slice(&result.names());
                denorm[idx] = Some(result);
            }
            if layer.categorical() {
                let (name, cl) = self.to_class(&som, som.weights(), idx).unwrap();
                classes[idx] = Some(cl);
                names.push(name);
            }
        }

        let mut writer = WriterBuilder::new()
            .delimiter(self.csv_options.delimiter)
            .from_path(path)?;

        let mut row = vec!["".to_string(); names.len()];
        writer.write_record(&names)?;
        for index in 0..som.weights().nrows() {
            let (r, c) = som.to_row_col(index);
            row[0] = index.to_string();
            row[1] = r.to_string();
            row[2] = c.to_string();

            for (idx, (layer, start_col)) in som
                .params()
                .layers()
                .iter()
                .zip(som.params().start_columns())
                .enumerate()
            {
                let mut offset_2 = 0;
                if class_values || !layer.categorical() {
                    let df = denorm[idx].as_ref().unwrap();
                    let df_row = df.get_row(index);
                    for i in 0..df_row.len() {
                        let v = df_row[i];
                        row[offset + *start_col + i] = v.to_string();
                    }
                    offset_2 += df_row.len()
                }

                if layer.categorical() {
                    let cls = classes[idx].as_ref().unwrap();
                    let v = &cls[index];
                    row[offset + offset_2 + *start_col] = v.clone();
                }
            }

            writer.write_record(&row)?;
        }

        Ok(())
    }

    /// Finds the nearest unit in the SOM for each row in `data`.
    ///
    /// # Returns
    /// A vector of (unit index, distance).
    pub fn nearest_unit(&self, som: &Som, data: &DataFrame) -> Vec<(usize, f64)> {
        assert_eq!(som.weights().names(), data.names());

        data.iter_rows()
            .map(|row| nearest_neighbor_xyf(row, som.weights(), self.layers()))
            .collect()
    }

    /// Writes `data`, amended by the nearest SOM unit index, row and column, to a CSV file.
    pub fn write_data_nearest(
        &self,
        som: &Som,
        data: &DataFrame,
        path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let mut classes: Vec<Option<Vec<String>>> = vec![None; self.layers.len()];
        let mut denorm: Vec<Option<DataFrame>> = (0..self.layers.len()).map(|_| None).collect();

        let nearest = self.nearest_unit(&som, data);

        let mut names: Vec<String> = self.preserve_columns.clone();
        let offset_preserved = names.len();

        for (idx, layer) in som.params().layers().iter().enumerate() {
            if layer.categorical() {
                let (name, cl) = self.to_class(&som, data, idx).unwrap();
                classes[idx] = Some(cl);
                names.push(name);
            } else {
                let result = self.to_denormalized(&som, data, idx).unwrap();
                names.extend_from_slice(&result.names());
                denorm[idx] = Some(result);
            }
        }

        let offset = names.len();
        names.extend_from_slice(&[
            "som_index".to_string(),
            "som_row".to_string(),
            "som_col".to_string(),
        ]);

        let no_data = &self.csv_options.no_data;

        let mut writer = WriterBuilder::new()
            .delimiter(self.csv_options.delimiter)
            .from_path(path)?;

        let mut row = vec!["".to_string(); names.len()];
        writer.write_record(&names)?;
        for index in 0..data.nrows() {
            for (idx, vec) in self.preserved.iter().enumerate() {
                row[idx] = vec[index].clone();
            }
            for (idx, (layer, start_col)) in som
                .params()
                .layers()
                .iter()
                .zip(som.params().start_columns())
                .enumerate()
            {
                if layer.categorical() {
                    let cls = classes[idx].as_ref().unwrap();
                    let v = &cls[index];
                    row[*start_col + offset_preserved] = v.clone();
                } else {
                    let df = denorm[idx].as_ref().unwrap();
                    let df_row = df.get_row(index);
                    for i in 0..df_row.len() {
                        let v = df_row[i];
                        row[*start_col + offset_preserved + i] = if v.is_nan() {
                            no_data.clone()
                        } else {
                            v.to_string()
                        };
                    }
                }
            }
            let (near, _dist) = nearest[index];
            let (r, c) = som.to_row_col(near);
            row[offset] = near.to_string();
            row[offset + 1] = r.to_string();
            row[offset + 2] = c.to_string();

            writer.write_record(&row)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::Neighborhood;
    use crate::calc::norm::Norm;
    use crate::map::som::DecayParam;
    use crate::proc::{InputLayer, ProcessorBuilder};

    #[test]
    fn nearest_unit() {
        let layers = vec![
            InputLayer::cont_simple(&[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]),
            InputLayer::cat_simple("species"),
        ];

        let proc = ProcessorBuilder::new(&layers, &vec![], &None, &None)
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

        let nearest = proc.nearest_unit(&som, proc.data());

        assert_eq!(nearest.len(), proc.data.nrows());

        //let result = proc.write_data_nearest(&som, proc.data(), "test.csv");
    }
    #[test]
    fn write_som() {
        let layers = vec![
            InputLayer::cont_simple(&[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]),
            InputLayer::cat_simple("species"),
        ];

        let proc = ProcessorBuilder::new(&layers, &vec![], &None, &None)
            .with_delimiter(b';')
            .build_from_file("example_data/iris.csv")
            .unwrap();

        let _som = proc.create_som(
            16,
            20,
            1000,
            Neighborhood::Gauss,
            DecayParam::lin(0.2, 0.01),
            DecayParam::lin(8.0, 0.5),
            DecayParam::exp(0.2, 0.001),
        );

        //let result = proc.write_som_units(&som, "test.csv", false);
    }
    #[test]
    fn layer_to_class() {
        let layers = vec![
            InputLayer::cont_simple(&[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]),
            InputLayer::cat_simple("species"),
        ];

        let proc = ProcessorBuilder::new(&layers, &vec![], &None, &None)
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
        let (name, classes) = proc.to_class(&som, som.weights(), 1).unwrap();
        assert_eq!(classes.len(), som.weights().nrows());
        assert_eq!(&name[..], "species");
    }

    #[test]
    fn denormalize_layer() {
        let layers = vec![
            InputLayer::cont_simple(&[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]),
            InputLayer::cat_simple("species"),
        ];

        let proc = ProcessorBuilder::new(&layers, &vec![], &None, &None)
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
        let denorm = proc.to_denormalized(&som, som.weights(), 0).unwrap();
        assert_eq!(denorm.nrows(), som.weights().nrows());
        assert_eq!(denorm.ncols(), proc.layers()[0].ncols());
    }

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

        let proc = ProcessorBuilder::new(&layers, &vec![], &None, &None)
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
    }
}
