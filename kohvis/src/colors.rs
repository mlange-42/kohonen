use gdnative::Color;

pub struct ColorPalette {
    colors: Vec<Color>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        ColorPalette {
            colors: vec![
                Color::rgb(1.0, 0.2, 0.2),
                Color::rgb(0.2, 0.2, 1.0),
                Color::rgb(0.2, 0.8, 0.2),
                Color::rgb(1.0, 0.0, 1.0),
                Color::rgb(0.0, 1.0, 1.0),
                Color::rgb(1.0, 1.0, 0.0),
                Color::rgb(0.6, 0.6, 0.6),
                Color::rgb(0.0, 0.7, 0.7),
                Color::rgb(0.9, 0.9, 0.9),
            ],
        }
    }
}

impl ColorPalette {
    pub fn get(&self, index: usize) -> &Color {
        &self.colors[index % self.colors.len()]
    }
}

pub trait ColorMap {
    fn get_color_norm(&self, value: f64) -> Color;
    fn get_color(&self, min: f64, max: f64, value: f64) -> Color {
        let range = max - min;
        self.get_color_norm((value - min) / range)
    }

    fn lerp(lower: f32, upper: f32, frac: f32) -> f32 {
        lower + frac * (upper - lower)
    }
    fn lerp_colors(lower: Color, upper: Color, frac: f32) -> Color {
        Color::rgb(
            Self::lerp(lower.r, upper.r, frac),
            Self::lerp(lower.g, upper.g, frac),
            Self::lerp(lower.b, upper.b, frac),
        )
    }
}

pub struct LinearColorMap {
    colors: Vec<Color>,
}
impl LinearColorMap {
    pub fn new(colors: &[&Color]) -> Self {
        LinearColorMap {
            colors: colors.iter().map(|c| **c).collect(),
        }
    }
}
impl ColorMap for LinearColorMap {
    fn get_color_norm(&self, value: f64) -> Color {
        let num_cols = self.colors.len();
        let rel = value * (num_cols - 1) as f64;
        let lower = rel.floor() as usize;
        let frac = rel - lower as f64;
        if frac < 0.001 {
            return self.colors[lower];
        }

        let col1 = self.colors[lower];
        let col2 = self.colors[lower + 1];
        Self::lerp_colors(col1, col2, frac as f32)
    }
}
