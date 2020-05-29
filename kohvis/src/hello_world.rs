
#[derive(gdnative::NativeClass)]
#[inherit(gdnative::Node)]
pub struct HelloWorld;

#[gdnative::methods]
impl HelloWorld {
    fn _init(_owner: gdnative::Node) -> Self {
        HelloWorld
    }
    
    #[export]
    fn _ready(&self, _owner: gdnative::Node) {
        godot_print!("hello, world.")
    }
}
