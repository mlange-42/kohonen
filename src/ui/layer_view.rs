use crate::calc::neighborhood::Neighborhood;
use crate::map::som::Som;
use easy_graph::color::style::{ShapeStyle, BLACK, GREEN, RED, WHITE, YELLOW};
use easy_graph::color::{ColorMap, LinearColorMap};
use easy_graph::ui::drawing::IntoDrawingArea;
use easy_graph::ui::element::Rectangle;
use easy_graph::ui::window::BufferWindow;

pub struct LayerView {
    window: BufferWindow,
    layers: Vec<usize>,
    layout_columns: Option<usize>,
}

impl LayerView {
    pub fn new(window: BufferWindow, layers: &[usize], layout_columns: Option<usize>) -> Self {
        LayerView {
            window,
            layers: layers.to_vec(),
            layout_columns,
        }
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn draw<N>(&mut self, som: &Som<N>)
    where
        N: Neighborhood,
    {
        let columns = self.get_columns(som);

        let margin = 5_i32;

        let (som_rows, som_cols) = som.size();
        let (width, height) = self.window.size();
        let width = width - 2 * margin as usize;
        let height = height - 2 * margin as usize;

        if self.layout_columns.is_none() {
            self.layout_columns = Some(Self::calc_layout_columns(
                width,
                height,
                som_rows,
                som_cols,
                columns.len(),
            ));
        }

        let layout_columns = self.layout_columns.unwrap();

        let layout_rows = (columns.len() as f64 / layout_columns as f64).ceil() as usize;
        let panel_width = width as f64 / layout_columns as f64;
        let panel_height = height as f64 / layout_rows as f64;

        let x_scale = panel_width / som_cols as f64;
        let y_scale = panel_height / som_rows as f64;
        let scale = (if x_scale < y_scale { x_scale } else { y_scale }) as i32;

        let ranges = som.weights().ranges();
        let color_map = LinearColorMap::new(&[&GREEN, &YELLOW, &RED]);

        self.window.draw(|b| {
            let root = b.into_drawing_area();
            root.fill(&WHITE).unwrap();
            for (index, col) in columns {
                let (v_min, v_max) = ranges[col];
                let lay_row = index / layout_columns;
                let lay_col = index % layout_columns;
                let x_min = margin + (lay_col as f64 * panel_width) as i32;
                let y_min = margin + (lay_row as f64 * panel_height) as i32;
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
            }
        });
    }

    fn get_columns<N>(&self, som: &Som<N>) -> Vec<(usize, usize)>
    where
        N: Neighborhood,
    {
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

    fn calc_layout_columns(
        width: usize,
        height: usize,
        som_rows: usize,
        som_cols: usize,
        data_columns: usize,
    ) -> usize {
        (1..data_columns)
            .map(|cols| {
                let layout_rows = (data_columns as f64 / cols as f64).ceil() as usize;
                let panel_width = width as f64 / cols as f64;
                let panel_height = height as f64 / layout_rows as f64;

                let x_scale = panel_width / som_cols as f64;
                let y_scale = panel_height / som_rows as f64;
                let scale = (if x_scale < y_scale { x_scale } else { y_scale }) as i32;

                (cols, scale)
            })
            .max_by(|(_col1, scale1), (_col2, scale2)| scale1.cmp(scale2))
            .unwrap()
            .0
    }
}

#[cfg(test)]
mod test {
    use crate::calc::neighborhood::GaussNeighborhood;
    use crate::map::som::{DecayParam, Layer, Som, SomParams};
    use crate::ui::layer_view::LayerView;
    use easy_graph::ui::window::WindowBuilder;

    #[test]
    fn view_layer() {
        let dim = 5;
        let params = SomParams::xyf(
            1000,
            GaussNeighborhood(),
            DecayParam::lin(0.1, 0.01),
            DecayParam::lin(10.0, 0.6),
            DecayParam::exp(0.25, 0.0001),
            vec![Layer::cont(3, 0.5), Layer::cat(2, 0.5)],
        );
        let som = Som::new(dim, 16, 20, params);

        let win = WindowBuilder::new()
            .with_dimensions(800, 600)
            .with_fps_skip(10.0)
            .build();

        let mut view = LayerView::new(win, &[0], None);

        //while view.window.is_open() {
        view.draw(&som);
        //}
    }
}
