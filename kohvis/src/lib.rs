#[macro_use]
extern crate gdnative;
extern crate kohonen;

mod colors;
mod kohonen_gd;
mod mapping_gd;
mod scatter3d;
mod tabs_gd;
pub mod util;
pub use kohonen_gd::Kohonen;
pub use mapping_gd::Mapping;
pub use scatter3d::Scatter3D;
pub use tabs_gd::Tabs;

fn init(handle: gdnative::init::InitHandle) {
    handle.add_class::<Kohonen>();
    handle.add_class::<Mapping>();
    handle.add_class::<Scatter3D>();
    handle.add_class::<Tabs>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
