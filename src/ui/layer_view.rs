//! Viewer for SOMs as heatmaps.

use crate::calc::nn::nearest_neighbor_xyf;
use crate::data::DataFrame;
use crate::map::som::Som;
use easy_graph::color::style::text_anchor::{HPos, Pos, VPos};
use easy_graph::color::style::{
    IntoFont, Palette, Palette99, ShapeStyle, TextStyle, BLACK, GREEN, RED, WHITE, YELLOW,
};
use easy_graph::color::{ColorMap, LinearColorMap};
use easy_graph::ui::drawing::IntoDrawingArea;
use easy_graph::ui::element::Rectangle;
use easy_graph::ui::window::BufferWindow;

/// Viewer for SOMs as heatmaps.
pub struct LayerView {
    window: BufferWindow,
    layers: Vec<usize>,
    names: Vec<String>,
    layout_columns: Option<usize>,
    scale: Option<i32>,
}

impl LayerView {
    /// Creates a new viewer for a selection of layers, or of all layers it `layers` is empty.
    pub fn new(
        window: BufferWindow,
        layers: &[usize],
        names: &[&str],
        layout_columns: Option<usize>,
    ) -> Self {
        LayerView {
            window,
            layers: layers.to_vec(),
            names: names.iter().map(|n| n.to_string()).collect(),
            layout_columns,
            scale: None,
        }
    }
    /// If the viewer's window is still open.
    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    /// Draws the given SOM. Should be called only for the same SOM repeatedly, not for different SOMs!
    pub fn draw(&mut self, som: &Som, data: Option<(&DataFrame, &[String])>) {
        let params = som.params();
        if (self.layers.len() == 1 && params.layers()[self.layers[0]].categorical())
            || (self.layers.is_empty()
                && params.layers().len() == 1
                && params.layers()[0].categorical())
        {
            self.draw_classes(som, data);
        } else {
            self.draw_columns(som);
        }
    }

    fn draw_classes(&mut self, som: &Som, data: Option<(&DataFrame, &[String])>) {
        let params = som.params();
        let layer = if self.layers.is_empty() {
            0
        } else {
            self.layers[0]
        };
        let start_col = params.start_columns()[layer];
        let classes: Vec<_> = self.names[start_col..(start_col + params.layers()[layer].ncols())]
            .iter()
            .map(|n| n.splitn(2, ':').nth(1).unwrap())
            .collect();

        let columns = self.get_columns(som);

        let margin = 5_i32;
        let heading = 16_i32;
        let legend = 120_i32;

        let (som_rows, som_cols) = som.size();
        let (width, height) = self.window.size();
        let width = width - 2 * margin as usize;
        let height = height - 2 * margin as usize;

        if self.layout_columns.is_none() {
            let (cols, scale) =
                Self::calc_layout_columns(width, height, som_rows, som_cols, 1, heading, legend);
            self.layout_columns = Some(cols);
            self.scale = Some(scale);
        }

        let scale = self.scale.unwrap();
        let test_style =
            TextStyle::from(("sans-serif", 14).into_font()).pos(Pos::new(HPos::Left, VPos::Top));
        let label_style = TextStyle::from(("sans-serif", 10).into_font())
            .pos(Pos::new(HPos::Center, VPos::Center));

        self.window.draw(|b| {
            let root = b.into_drawing_area();
            root.fill(&WHITE).unwrap();

            let x_min = margin;
            let y_min = margin + heading;

            // Draw units
            for (idx, row) in som.weights().iter_rows().enumerate() {
                let (r, c) = som.to_row_col(idx);
                let x = x_min + (c as i32 * scale);
                let y = y_min + (r as i32 * scale);

                let mut v_max = std::f64::MIN;
                let mut idx_max = 0;
                for (index, col) in columns.iter() {
                    let v = row[*col];
                    if v > v_max {
                        v_max = v;
                        idx_max = *index;
                    }
                }

                let color = Palette99::pick(idx_max); //color_map.get_color(v_min, v_max, v);

                root.draw(&Rectangle::new(
                    [(x, y), (x + scale, y + scale)],
                    ShapeStyle::from(&color).filled(),
                ))
                .unwrap();
            }

            // Draw outline
            root.draw(&Rectangle::new(
                [
                    (x_min, y_min),
                    (
                        x_min + scale * som_cols as i32,
                        y_min + scale * som_rows as i32,
                    ),
                ],
                ShapeStyle::from(&BLACK),
            ))
            .unwrap();

            // Draw labels
            if let Some((data, labels)) = data {
                let nearest: Vec<_> = data
                    .iter_rows()
                    .map(|row| nearest_neighbor_xyf(row, som.weights(), som.params().layers()))
                    .collect();
                let mut total_counts = vec![0; som.weights().nrows()];
                let mut counts = vec![0; som.weights().nrows()];
                for (idx, _) in &nearest {
                    total_counts[*idx] += 1;
                }
                for ((idx, _), label) in nearest.iter().zip(labels) {
                    let (r, c) = som.to_row_col(*idx);
                    let offset = 1.0 / (total_counts[*idx] + 1) as f64;
                    let x = x_min + (c as i32 * scale) + (0.5 * scale as f64) as i32;
                    let y = y_min
                        + (r as i32 * scale)
                        + (offset * (counts[*idx] + 1) as f64 * scale as f64) as i32;
                    root.draw_text(&label, &label_style, (x, y)).unwrap();

                    counts[*idx] += 1;
                }
            }

            // Draw lagend
            let x = x_min + som.ncols() as i32 * scale + 10;
            for (i, class) in classes.iter().enumerate() {
                let color = Palette99::pick(i);
                root.draw(&Rectangle::new(
                    [
                        (x, y_min + i as i32 * 14),
                        (x + 10, y_min + i as i32 * 14 + 10),
                    ],
                    ShapeStyle::from(&color).filled(),
                ))
                .unwrap();
                root.draw_text(class, &test_style, (x + 14, y_min + i as i32 * 14))
                    .unwrap();
            }
        });
    }

    fn draw_columns(&mut self, som: &Som) {
        let columns = self.get_columns(som);

        let margin = 5_i32;
        let heading = 16_i32;
        let legend = 20_i32;

        let (som_rows, som_cols) = som.size();
        let (width, height) = self.window.size();
        let width = width - 2 * margin as usize;
        let height = height - 2 * margin as usize;

        if self.layout_columns.is_none() {
            let (cols, scale) = Self::calc_layout_columns(
                width,
                height,
                som_rows,
                som_cols,
                columns.len(),
                heading,
                legend,
            );
            self.layout_columns = Some(cols);
            self.scale = Some(scale);
        }

        let layout_columns = self.layout_columns.unwrap();

        let layout_rows = (columns.len() as f64 / layout_columns as f64).ceil() as usize;
        let panel_width = width as f64 / layout_columns as f64;
        let panel_height = height as f64 / layout_rows as f64;

        let scale = self.scale.unwrap();

        let ranges = som.weights().ranges();

        let color_map = LinearColorMap::new(&[&GREEN, &YELLOW, &RED]);
        let names = &self.names;
        let test_style =
            TextStyle::from(("sans-serif", 14).into_font()).pos(Pos::new(HPos::Left, VPos::Bottom));

        self.window.draw(|b| {
            let root = b.into_drawing_area();
            root.fill(&WHITE).unwrap();
            for (index, col) in columns {
                let (v_min, v_max) = ranges[col];
                let lay_row = index / layout_columns;
                let lay_col = index % layout_columns;
                let x_min = margin + (lay_col as f64 * panel_width) as i32;
                let y_min = margin + heading + (lay_row as f64 * panel_height) as i32;
                for (idx, row) in som.weights().iter_rows().enumerate() {
                    let (r, c) = som.to_row_col(idx);
                    let v = row[col];
                    let x = x_min + (c as i32 * scale);
                    let y = y_min + (r as i32 * scale);

                    let color = color_map.get_color(v_min, v_max, v);

                    root.draw(&Rectangle::new(
                        [(x, y), (x + scale, y + scale)],
                        ShapeStyle::from(&color).filled(),
                    ))
                    .unwrap();
                }
                root.draw(&Rectangle::new(
                    [
                        (x_min, y_min),
                        (
                            x_min + scale * som_cols as i32,
                            y_min + scale * som_rows as i32,
                        ),
                    ],
                    ShapeStyle::from(&BLACK),
                ))
                .unwrap();
                root.draw_text(&names[col], &test_style, (x_min, y_min - 1))
                    .unwrap();
                let steps = 25;
                let total_height = scale * som.nrows() as i32 - 40;
                let total_width = scale * som.ncols() as i32;
                let x = x_min + total_width;
                for i in 0..steps {
                    let value = i as f64 / steps as f64;
                    let color = color_map.get_color(0.0, 1.0, value);
                    let y = y_min + total_height + 20 - (total_height as f64 * value) as i32;
                    root.draw(&Rectangle::new(
                        [
                            (x + 3, y),
                            (
                                x + legend - 3,
                                y + (total_height as f64 / steps as f64) as i32,
                            ),
                        ],
                        ShapeStyle::from(&color).filled(),
                    ))
                    .unwrap();
                }
            }
        });
    }

    /// Calculates the required columns as a vector of (index, column index).
    fn get_columns(&self, som: &Som) -> Vec<(usize, usize)> {
        let params = som.params();
        let mut columns = vec![];
        if params.layers().is_empty() || self.layers.is_empty() {
            columns.extend((0..som.weights().ncols()).map(|c| (c, c)));
        } else {
            let mut start = 0;
            let mut index = 0;
            for (l, layer) in params.layers().iter().enumerate() {
                if self.layers.contains(&l) {
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
        width: usize,
        height: usize,
        som_rows: usize,
        som_cols: usize,
        data_columns: usize,
        heading: i32,
        legend: i32,
    ) -> (usize, i32) {
        if data_columns == 1 {
            let panel_width = width as f64 - legend as f64;
            let panel_height = height as f64 - heading as f64;

            let x_scale = panel_width / som_cols as f64;
            let y_scale = panel_height / som_rows as f64;
            let scale = (if x_scale < y_scale { x_scale } else { y_scale }) as i32;

            (1, scale)
        } else {
            (1..data_columns)
                .map(|cols| {
                    let layout_rows = (data_columns as f64 / cols as f64).ceil() as usize;
                    let panel_width = (width as f64 / cols as f64) - legend as f64;
                    let panel_height = (height as f64 / layout_rows as f64) - heading as f64;

                    let x_scale = panel_width / som_cols as f64;
                    let y_scale = panel_height / som_rows as f64;
                    let scale = (if x_scale < y_scale { x_scale } else { y_scale }) as i32;

                    (cols, scale)
                })
                .max_by(|(_col1, scale1), (_col2, scale2)| scale1.cmp(scale2))
                .unwrap()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::Neighborhood;
    use crate::map::som::{DecayParam, Layer, Som, SomParams};
    use crate::ui::layer_view::LayerView;
    use easy_graph::ui::window::WindowBuilder;

    #[test]
    fn view_layer() {
        let cols = ["A", "B", "C", "D", "E"];
        let params = SomParams::xyf(
            1000,
            Neighborhood::Gauss,
            DecayParam::lin(0.1, 0.01),
            DecayParam::lin(10.0, 0.6),
            DecayParam::exp(0.25, 0.0001),
            vec![Layer::cont(3, 0.5), Layer::cat(2, 0.5)],
        );
        let som = Som::new(&cols, 16, 20, params);

        let win = WindowBuilder::new()
            .with_dimensions(800, 600)
            .with_fps_skip(10.0)
            .build();

        let mut view = LayerView::new(win, &[0], &cols, None);

        //while view.window.is_open() {
        view.draw(&som, None);
        //}
    }
}
