use crate::KohonenUser2D;
use gdnative::{Color, Control, Vector2};

#[derive(gdnative::NativeClass)]
#[inherit(gdnative::Control)]
pub struct Mapping {
    #[property()]
    kohonen_path: String,
}

#[gdnative::methods]
impl Mapping {
    fn _init(_owner: gdnative::Control) -> Self {
        Mapping {
            kohonen_path: "".to_string(),
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
        owner.draw_circle(Vector2::new(10., 10.), 25., Color::rgb(1.0, 0.0, 0.0));
        godot_print!("Drawing");
    }
}

impl KohonenUser2D for Mapping {
    fn kohonen_path(&self) -> &str {
        &self.kohonen_path
    }
}
