#[macro_use]
extern crate gdnative;
extern crate kohonen;

mod kohonen_gd;
pub use kohonen_gd::Kohonen;

fn init(handle: gdnative::init::InitHandle) {
    handle.add_class::<Kohonen>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
