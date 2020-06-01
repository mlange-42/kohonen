extends Spatial


onready var arm = $CameraArm
onready var camera = $CameraArm/Camera as Camera

const MOUSE_SENSITIVITY = 0.25
const PAN_SENSITIVITY = 0.005

func _ready():
	pass # Replace with function body.

func _input(event):
	if event is InputEventMouseMotion:
		if Input.is_mouse_button_pressed(BUTTON_LEFT):
			arm.rotate_x(deg2rad(event.relative.y * MOUSE_SENSITIVITY * -1))
			self.rotate_y(deg2rad(event.relative.x * MOUSE_SENSITIVITY * -1))
	
			var rotation = arm.rotation_degrees
			rotation.x = clamp(rotation.x, -80, 80)
			arm.rotation_degrees = rotation
		elif Input.is_mouse_button_pressed(BUTTON_MIDDLE):
			var x = arm.transform.basis.x * event.relative.x * PAN_SENSITIVITY * -1
			var y = arm.transform.basis.y * event.relative.y * PAN_SENSITIVITY
			translate(x + y)
		
	if event is InputEventMouseButton and event.is_pressed():
		if event.button_index == BUTTON_WHEEL_UP:
			camera.translate_object_local(Vector3(0, 0, 0.1))
		if event.button_index == BUTTON_WHEEL_DOWN:
			camera.translate_object_local(Vector3(0, 0, -0.1))
