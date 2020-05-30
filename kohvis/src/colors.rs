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
