#[macro_use]
extern crate gdnative;

mod hello_world;
pub use hello_world::HelloWorld;

fn init(handle: gdnative::init::InitHandle) {
    handle.add_class::<HelloWorld>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
