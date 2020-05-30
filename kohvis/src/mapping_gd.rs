use crate::colors::{ColorMap, ColorPalette, LinearColorMap};
use crate::{Kohonen, KohonenUser2D};
use gdnative::{Color, Control, Font, GodotString, Int32Array, ResourceLoader};
use kohonen::calc::nn::nearest_neighbor_xyf;
use kohonen::data::DataFrame;
use kohonen::map::som::Som;

#[derive(gdnative::NativeClass)]
#[inherit(gdnative::Control)]
pub struct Mapping {
    #[property()]
    kohonen_path: String,
    #[property()]
    layers: Int32Array,
    #[property()]
    layout_columns: Option<i32>,
    scale: Option<f32>,
    colors: ColorPalette,
    font: Font,
}

#[gdnative::methods]
impl Mapping {
    fn _init(_owner: gdnative::Control) -> Self {
        let mut layers = Int32Array::new();
        layers.push(0);
        let mut loader = ResourceLoader::godot_singleton();
        let font = ResourceLoader::load(
            &mut loader,
            GodotString::from_str("res://fonts/arial.tres"),
            GodotString::from_str(""),
            false,
        )
        .and_then(|s| s.cast::<Font>())
        .unwrap();
        Mapping {
            kohonen_path: "".to_string(),
            layers,
            layout_columns: None,
            scale: None,
            colors: ColorPalette::default(),
            font,
        }
    }

    #[export]
    fn _ready(&mut self, _owner: Control) {}

    #[export]
    fn _process(&mut self, owner: Control, _delta: f64) {
        self.update(owner);
    }

    #[export]
    unsafe fn _draw(&mut self, mut owner: Control) {
        owner.set_clip_contents(true);
        let mut layers = Vec::new();
        for i in 0..self.layers.len() {
            layers.push(self.layers.get(i));
        }

        Self::with_kohonen(
            owner,
            &self.kohonen_path().to_string(),
            |owner: &mut gdnative::Control, koh: &Kohonen| {
                let proc = koh.processor().as_ref().unwrap();
                let label_data = match proc.labels() {
                    Some(lab) => Some((proc.data(), lab)),
                    None => None,
                };
                let som = koh.som().as_ref().unwrap();
                let params = som.params();
                if (layers.len() == 1 && params.layers()[layers[0] as usize].categorical())
                    || (layers.len() == 0
                        && params.layers().len() == 1
                        && params.layers()[0].categorical())
                {
                    let margin = 5_i32;
                    let heading = 16_i32;
                    let legend = 120_i32;

                    let (som_rows, som_cols) = som.size();
                    let control_size = owner.get_size();
                    let (width, height) = (control_size.x, control_size.y);
                    let width = width - 2. * margin as f32;
                    let height = height - 2. * margin as f32;

                    let (cols, scale) = Self::calc_layout_columns(
                        width, height, som_rows, som_cols, 1, heading, legend,
                    );
                    self.layout_columns = Some(cols as i32);
                    self.scale = Some(scale);

                    let names = proc.data().columns();

                    self.draw_classes(owner, som, label_data, names);
                } else {
                    let names = proc.data().columns();
                    self.draw_columns(owner, som, names);
                }
            },
        );
    }

    fn draw_classes(
        &self,
        owner: &mut Control,
        som: &Som,
        data: Option<(&DataFrame, &[(usize, String)])>,
        names: &[String],
    ) {
        let params = som.params();
        let layer = if self.layers.len() == 0 {
            0
        } else {
            self.layers.get(0) as usize
        };
        let start_col = params.start_columns()[layer];
        let classes: Vec<_> = names[start_col..(start_col + params.layers()[layer].ncols())]
            .iter()
            .map(|n| n.splitn(2, ':').nth(1).unwrap())
            .collect();

        let columns = self.get_columns(som);

        let margin = 5_i32;
        let heading = 16_i32;

        let scale = self.scale.unwrap();

        let x_min = margin;
        let y_min = margin + heading;

        let black = Color::rgb(0., 0., 0.);
        let white = Color::rgb(1., 1., 1.);

        // Draw units
        for (idx, row) in som.weights().iter_rows().enumerate() {
            let (r, c) = som.to_row_col(idx);
            let x = x_min as f32 + (c as f32 * scale);
            let y = y_min as f32 + (r as f32 * scale);

            let mut v_max = std::f64::MIN;
            let mut idx_max = 0;
            for (index, col) in columns.iter() {
                let val = row[*col];
                if val > v_max {
                    v_max = val;
                    idx_max = *index;
                }
            }

            let color = self.colors.get(idx_max).clone();

            unsafe {
                owner.draw_rect(
                    euclid::Rect::new(
                        euclid::Point2D::new(x, y),
                        euclid::Size2D::new(scale, scale),
                    ),
                    color,
                    true,
                    1.0,
                    false,
                );
            }
        }

        // Draw labels
        if let Some((data, labels)) = data {
            let nearest: Vec<_> = labels
                .iter()
                .map(|(idx, _lab)| {
                    nearest_neighbor_xyf(data.get_row(*idx), som.weights(), som.params().layers())
                })
                .collect();

            let mut total_counts = vec![0; som.weights().nrows()];
            let mut counts = vec![0; som.weights().nrows()];
            for (idx, _) in &nearest {
                total_counts[*idx] += 1;
            }
            for ((idx, _), (_data_idx, label)) in nearest.iter().zip(labels) {
                let (r, c) = som.to_row_col(*idx);
                let offset = 1.0 / (total_counts[*idx] + 1) as f64;
                let x = x_min as f32 + (c as f32 * scale) + (0.5 * scale as f64) as f32;
                let y = y_min as f32
                    + (r as f32 * scale)
                    + (offset * (counts[*idx] + 1) as f64 * scale as f64) as f32;

                let text = GodotString::from_str(label);
                let size = self.font.get_string_size(text.clone());
                unsafe {
                    owner.draw_string(
                        Some(self.font.clone()),
                        euclid::Vector2D::new((x - size.x / 2.0).round(), y.round()),
                        text,
                        black,
                        -1,
                    );
                }

                counts[*idx] += 1;
            }
        }

        // Draw legend
        let x = x_min as f32 + som.ncols() as f32 * scale + 10.;
        for (i, class) in classes.iter().enumerate() {
            let color = self.colors.get(i).clone();
            unsafe {
                owner.draw_rect(
                    euclid::Rect::new(
                        euclid::Point2D::new(x, y_min as f32 + i as f32 * 14.),
                        euclid::Size2D::new(10., 10.),
                    ),
                    color,
                    true,
                    1.0,
                    false,
                );
                owner.draw_string(
                    Some(self.font.clone()),
                    euclid::Vector2D::new(
                        (x + 14.).round(),
                        (y_min as f32 + i as f32 * 14. + 10.).round(),
                    ),
                    GodotString::from_str(class),
                    white,
                    -1,
                );
            }
        }
    }

    fn draw_columns(&mut self, owner: &mut Control, som: &Som, names: &[String]) {
        let columns = self.get_columns(som);

        let margin = 5_i32;
        let heading = 16_i32;
        let legend = 20_i32;

        let (som_rows, som_cols) = som.size();
        let control_size = unsafe { owner.get_size() };
        let (width, height) = (control_size.x, control_size.y);
        let width = width - 2. * margin as f32;
        let height = height - 2. * margin as f32;

        let (cols, scale) = Self::calc_layout_columns(
            width,
            height,
            som_rows,
            som_cols,
            columns.len(),
            heading,
            legend,
        );
        self.layout_columns = Some(cols as i32);
        self.scale = Some(scale);

        let layout_columns = self.layout_columns.unwrap();

        let layout_rows = (columns.len() as f64 / layout_columns as f64).ceil() as usize;
        let panel_width = width as f64 / layout_columns as f64;
        let panel_height = height as f64 / layout_rows as f64;

        let scale = self.scale.unwrap();

        let ranges = som.weights().ranges();

        let color_map = LinearColorMap::new(&[
            &Color::rgb(0.7, 0.0, 0.65),
            &Color::rgb(1.0, 0.0, 0.0),
            &Color::rgb(1.0, 1.0, 0.0),
            &Color::rgb(0.0, 1.0, 0.0),
            &Color::rgb(0.0, 1.0, 1.0),
        ]);
        let black = Color::rgb(0., 0., 0.);
        let white = Color::rgb(1., 1., 1.);

        for (index, col) in columns {
            let (v_min, v_max) = ranges[col];
            let lay_row = index / self.layout_columns.unwrap() as usize;
            let lay_col = index % self.layout_columns.unwrap() as usize;
            let x_min = margin as f32 + (lay_col as f64 * panel_width) as f32;
            let y_min = margin as f32 + heading as f32 + (lay_row as f64 * panel_height) as f32;
            for (idx, row) in som.weights().iter_rows().enumerate() {
                let (r, c) = som.to_row_col(idx);
                let val = row[col];
                let x = x_min as f32 + (c as f32 * scale);
                let y = y_min as f32 + (r as f32 * scale);

                let color = color_map.get_color(v_min, v_max, val);

                unsafe {
                    owner.draw_rect(
                        euclid::Rect::new(
                            euclid::Point2D::new(x, y),
                            euclid::Size2D::new(scale, scale),
                        ),
                        color,
                        true,
                        1.0,
                        false,
                    );
                }
            }
            unsafe {
                owner.draw_rect(
                    euclid::Rect::new(
                        euclid::Point2D::new(x_min, y_min),
                        euclid::Size2D::new(scale * som_cols as f32, scale * som_rows as f32),
                    ),
                    black,
                    false,
                    1.0,
                    false,
                );
            }
            let text = GodotString::from_str(&names[col]);

            unsafe {
                owner.draw_string(
                    Some(self.font.clone()),
                    euclid::Vector2D::new(x_min.round(), y_min.round() - 2.),
                    text,
                    white,
                    -1,
                );
            }

            let steps = 25;
            let total_height = scale * som.nrows() as f32 - 40.;
            let total_width = scale * som.ncols() as f32;
            let x = x_min as f32 + total_width;
            for i in 0..steps {
                let value = i as f64 / steps as f64;
                let color = color_map.get_color(0.0, 1.0, value);
                let y = y_min as f32 + total_height + 20. - (total_height as f32 * value as f32);
                unsafe {
                    owner.draw_rect(
                        euclid::Rect::new(
                            euclid::Point2D::new(x + 3., y),
                            euclid::Size2D::new(legend as f32 - 3., total_height / steps as f32),
                        ),
                        color,
                        true,
                        1.0,
                        false,
                    );
                }
            }
        }
    }

    /// Calculates the required columns as a vector of (index, column index).
    fn get_columns(&self, som: &Som) -> Vec<(usize, usize)> {
        let params = som.params();
        let mut columns = vec![];
        if params.layers().is_empty() || self.layers.len() == 0 {
            columns.extend((0..som.weights().ncols()).map(|c| (c, c)));
        } else {
            let mut start = 0;
            let mut index = 0;
            for (l, layer) in params.layers().iter().enumerate() {
                let mut contain = false;
                for i in 0..self.layers.len() {
                    if self.layers.get(i) == l as i32 {
                        contain = true;
                        break;
                    }
                }
                if contain {
                    for i in 0..layer.ncols() {
                        columns.push((index, start + i));
                        index += 1;
                    }
                }
                start += layer.ncols();
            }
        };
        columns
    }
    /// Calculates the optimum number of layout columns.
    fn calc_layout_columns(
        width: f32,
        height: f32,
        som_rows: usize,
        som_cols: usize,
        data_columns: usize,
        heading: i32,
        legend: i32,
    ) -> (usize, f32) {
        if data_columns == 1 {
            let panel_width = width as f64 - legend as f64;
            let panel_height = height as f64 - heading as f64;

            let x_scale = panel_width / som_cols as f64;
            let y_scale = panel_height / som_rows as f64;
            let scale = if x_scale < y_scale { x_scale } else { y_scale };

            (1, scale as f32)
        } else {
            (1..data_columns)
                .map(|cols| {
                    let layout_rows = (data_columns as f64 / cols as f64).ceil() as usize;
                    let panel_width = (width as f64 / cols as f64) - legend as f64;
                    let panel_height = (height as f64 / layout_rows as f64) - heading as f64;

                    let x_scale = panel_width / som_cols as f64;
                    let y_scale = panel_height / som_rows as f64;
                    let scale = if x_scale < y_scale { x_scale } else { y_scale };

                    (cols, scale as f32)
                })
                .max_by(|(_col1, scale1), (_col2, scale2)| scale1.partial_cmp(scale2).unwrap())
                .unwrap()
        }
    }
}

impl KohonenUser2D for Mapping {
    fn kohonen_path(&self) -> &str {
        &self.kohonen_path
    }
}
