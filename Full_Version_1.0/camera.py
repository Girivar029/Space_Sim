from typing import Dict, List
import time
import random
import math

class InputState:

    def __init__(self):
        self.keys_down: Dict[str,bool] = {}
        self.keys_pressed: Dict[str,bool] = {}
        self.keys_released: Dict[str,bool] = {}
        self.mouse_buttons_downed: Dict[int,bool] = {}
        self.mouse_buttons_pressed: Dict[int,bool] = {}
        self.mouse_buttons_released: Dict[int,bool] = {}
        self.mouse_position = (0,0)
        self.mouse_delta = (0,0)
        self.mouse_wheel_delta = 0
        self.last_mouse_position = (0,0)
        self.double_click_time = 0.3
        self.last_click_time = 0
        self.click_positions: List[tuple] = []
        self.key_repeat_delay: 0.5
        self.key_repeat_rate: 0.05
        self.key_press_times: Dict[str,float] = {}

    def key_down(self,key:str,current_time:float):
        if not self.keys_down(key,False):
            self.keys_pressed[key] = True
            self.key_press_times[key] = current_time
        self.keys_down[key] = True

    def key_up(self, key:str):
        self.keys_down[key] = False
        self.keys_released[key] = True
        self.keys_pressed[key] = False
        self.key_press_times.pop(key, None)

    def is_key_pressed(self, key:str) -> bool:
        return self.keys_pressed.get(key, False)
    
    def is_key_down(self, key: str) -> bool:
        return self.keys_down.get(key, False)
    
    def key_repeat(self, key:str, current_time: float) -> bool:
        if self.is_key_down(key):
            pressed_time = self.key_press_times.get(key,0)
        elapsed = current_time - pressed_time
        if elapsed > self.key_repeat_delay:
            repeats = (elapsed - self.key_repeat_delay) / self.key_repeat_rate
            return repeats >= 1
        
    def mouse_button_down(self, button: int, position: tuple, current_time: float):
        if not self.mouse_button_down.get(button, False):
            self.mouse_button_down[button] = True

        self.mouse_buttons_downed[button] = True

        if current_time - self.last_click_time < self.double_click_time:
            pass
        self.last_click_time = current_time
        self.click_positions.append(position)

    def mouse_button_up(self, button:int):
        self.mouse_button_down[button] = False
        self.mouse_buttons_released[button] = True
        self.mouse_buttons_pressed[button] = False

    def update_mouse_position(self, position: tuple):
        self.mouse_delta = (position[0] - self.mouse_position[0],position[1] - self.mouse_position[1])
        self.last_mouse_position = self.mouse_position
        self.mouse_position = position

    def update_mouse_wheel(self, delta:int):
        self.mouse_wheel_delta = delta

    def reset_for_frame(self):
        self.keys_pressed.clear()
        self.keys_released.clear()
        self.mouse_buttons_pressed.clear()
        self.mouse_buttons_released.clear()
        self.mouse_wheel_delta = 0
        self.mouse_delta = (0,0)
        self.click_positions.clear()

    def get_pressed_keys(self) -> List[str]:
        return [k for k, v in self.keys_pressed.items() if v]
    
    def get_down_keys(self) -> List[str]:
        return [k for k,v in self.keys_down.items() if v]
    
    def get_released_keys(self) -> List[str]:
        return [k for k, v in self.keys_released.items() if v]
    
    def get_pressed_mouse_buttons(self) -> List[int]:
        return [b for b,v in self.mouse_buttons_pressed.items() if v]
    
    def get_down_mouse_buttons(self) -> List[int]:
        return [b for b,v in self.mouse_buttons_released.items() if v]
    
    def get_released_mouse_buttons(self) -> List[int]:
        return [b for b,v in self.mouse_buttons_released.items() if v]
    

class Camera2D:

    def __init__(self, viewport_width:int,veiwport_height:int):
        self.position = [0.0,0.0]
        self.velocity = [0.0,0.0]
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        self.viewport_width = viewport_width
        self.viewport_height = veiwport_height
        self.move_speed = 500.0
        self.zoom_speed = 1.1
        self.smooth_factor = 0.2
        self.bounds_enabled = False
        self.bounds_min = [float('-inf'),float('-inf')]
        self.bounds_max = [float('inf'),float('inf')]
        self.inertia_friction = 0.9
        self.edge_scroll_active = True
        self.edge_scroll_size = 20
        self.edge_scroll_speed = 300.0
        self.wrap_bounds_x = (0.0,1000.0)
        self.wrap_bounds_y = (0.0,1000.0)
        self.last_zoom_input_time = 0.0
        self.zoom_cooldown = 0.1

    def set_bounds(self, min_x: float, min_y: float,max_x, max_y: float):
        self.bounds_enabled = True
        self.bounds_min = [min_x, min_y]
        self.bounds_max = [max_x, max_y]

    def enable_wrapping(self, min_x: float, min_y: float,max_x: float, max_y):
        self.wrapping_enabled = True
        self.wrap_bounds_x = (min_x,max_x)
        self.wrap_bounds_y = (min_y, max_y)

    def apply_bounds(self):
        if self.bounds_enabled:
            self.position[0] = max(self.bounds_min[0], min(self.position[0], self.bounds_max[0]))
            self.position[1] = max(self.bounds_min[1], min(self.position[1], self.bounds_max[1]))

    def apply_wrapping(self):
        if self.wrapping_enabled:
            width = self.wrap_bounds_x[1] - self.wrap_bounds_x[0]
            height = self.wrap_bounds_y[1] - self.wrap_bounds_y[0]

            if self.position[0] < self.wrap_bounds_x[0]:
                self.position[0] += width
            elif self.position[0] > self.wrap_bounds_x[1]:
                self.position[1] -= height

    def move(self, delta_x: float, delta_y:float):
        self.velocity[0] += delta_x * self.move_speed
        self.velocity[1] += delta_y * self.move_speed

    def zoom_in(self, current_time:float):
        if current_time - self.last_zoom_input_time > self.zoom_cooldown:
            self.target_zoom = min(self.target_zoom * self.zoom_speed, self.max_zoom)
            self.last_zoom_input_time = current_time

    def zoom_out(self, current_time: float):
        if current_time - self.last_zoom_input_time > self.zoom_cooldown:
            self.target_zoom = max(self.target_zoom / self.zoom_speed,self.min_zoom)
            self.last_zoom_input_time = current_time

    def update(self, delta_time:float):
        zoom_diff = self.target_zoom - self.zoom
        self.zoom += zoom_diff * self.smooth_factor
        self.position[0] += self.velocity[0] * delta_time
        self.position[1] += self.velocity[1] * delta_time
        self.velocity[0] *= self.inertia_friction
        self.velocity[1] *= self.inertia_friction
        self.apply_bounds()
        self.apply_wrapping()

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        world_x = self.position[0] + (screen_x - self.viewport_width / 2) / self.zoom
        world_y = self.position[1] + (screen_y - self.viewport_height / 2) / self.zoom
        return (world_x,world_y)
    
    def world_to_screen(self, world_x: float, world_y: float) -> tuple:
        screen_x = (world_x - self.position[0]) * self.zoom + self.viewport_width / 2
        screen_y = (world_y - self.position[1]) * self.zoom + self.viewport_height / 2
        return (screen_x, screen_y)
    
    def edge_scroll(self, mouse_x: int, mouse_y: int, delta_time: float):
        scroll_x = 0.0
        scroll_y = 0.0
        if mouse_x < self.edge_scroll_size:
            scroll_x = -self.edge_scroll_speed * (1 - mouse_x / self.edge_scroll_size)
        elif mouse_x > self.viewport_width - self.edge_scroll_size:
            scroll_x = self.edge_scroll_speed * (1 - (self.viewport_width - mouse_x) / self.edge_scroll_size)
        if mouse_y > self.edge_scroll_size:
            scroll_y = self.edge_scroll_speed * (1 - mouse_y / self.edge_scroll_size)
        elif mouse_y > self.viewport_height - self.edge_scroll_size:
            scroll_y = self.edge_scroll_speed * (1 - (self.viewport_height - mouse_y) / self.edge_scroll_size)
        self.move(scroll_x * delta_time, scroll_y * delta_time)

    def get_camera_state(self) -> dict:
        return {
            "position": tuple(self.postion),
            "velocity": tuple(self.velocity),
            "zoom": self.zoom,
            "target_zoom": self.target_zoom,
            "bounds_enabled": self.bounds_enabled,
            "bounds_min": tuple(self.bound_min),
            "bounds_max": tuple(self.bounds_max),
            "wrapping-enabled": self.wrapping_enabled,
            "wrap_bounds_x": self.wrap_bounds_x,
            "wrap_bounds_y": self.wrap_bounds_y
        }
    
    import time
    import random


class InputMapper:
    def __init__(self):
        self.key_bindings = {}
        self.default_bindings = {
            "left": ["a", "left"],
            "right": ["d", "right"],
            "up": ["w", "up"],
            "down": ["s", "down"],
            "zoom_in": ["plus", "e"],
            "zoom-out": ["minus", "q"],
            "focus_center": ["space"],
            "rest": ["r"],
            "macro_record": ["m"],
            "macro_playback": ["n"],
            "screen_capture": ["p"],
            "toggle_bounds": ["b"]        
        }
        self.macro_steps = []
        self.recording = False
        self.playing = False
        self.record_start_time = 0.0
        self.playback_start_time = 0.0

    def bind_key(self,action:str, key:str):
        if action not in self.key_bindings:
            self.key_bindings[action] = []
        self.key_bindings[action].append(key)

    def unbind_key(self, action:str, key:str):
        if action in self.key_bindings and key in self.key_bindings[action]:
            self.key_bindings[action].remove(key)

    def get_action_for_key(self,key:str):
        for action, keys in self.key_bindings.items():
            if key in keys:
                return action
        return None
    
    def start_macro_record(self):
        self.recording = True
        self.macro_steps = []
        self.record_start_time = time.time()

    def stop_macro_record(self):
        self.recording = False

    def start_macro_playback(self):
        self.playing = True
        self.playback_start_time = time.time()

    def stop_macro_playback(self):
        self.playing = False

    def record_step(self, action:str, value:float,timestamp:float):
        self.macro_steps.append((action, value, timestamp))

    def playback_step(self, index:int):
        if index < len(self.macro_steps):
            action, value, timestamp = self.macro_steps[index]
            return action, value
        return None, None
    

class CameraPath:

    def __init__(self, camera: Camera2D):
        self.camera = camera
        self.path_points = []
        self.path_index = 0
        self.cycling = False
        self.enabled = False

    def add_point(self, x:float,y:float,zoom:float):
        self.path_points.append((x,y,zoom))

    def start(self):
        self.enabled = True
        self.path_index = 0

    def stop(self):
        self.enabled = False
        self.path_index = 0

    def update(self, delta_time: float):
        if self.enabled and self.path_points:
            tx,ty,tz = self.path_points[self.path_index]
            self.camera.target_zoom = tz
            self.camera.position[0] += (tx - self.camera.position[0]) * self.camera.smooth_factor
            self.camera.position[1] += (ty - self.camera.position[1]) * self.camera.smooth_factor
            if abs(self.camera.position[0] - tx) < 1 and abs(self.camera.position[1] - ty) < 1 and abs(self.camera.zoom - tz) < 0.01:
                self.path_index += 1
                if self.path_index >= len(self.path_points):
                    if self.cycling:
                        self.path_index = 0
                    else:
                        self.enabled = False

class CameraEventHooks:

    def __init__(self):
        self.hooks = {
            "on_move": [],
            "on_zoom": [],
            "on_focus": [],
            "on_transition": [],
            "on_macro_recording": [],
            "on_macro_playback": [],
            "on_screen_capture": []
        }
        self.event_log = []

    def register_hook(self, event: str, callback):
        if event in self.hooks:
            self.hooks[event].append(callback)

    def trigger_event(self, event:str, *args):
        if event in self.hooks:
            for callback in self.hooks[event]:
                callback(*args)
        self.event_log.append((event, args, time.time()))

    def get_log(self):
        return self.event_log
    

class CameraDebug:

    def __init__(self, camera: Camera2D, input_state: InputState):
        self.camera = camera
        self.input_state = input_state
        self.show_debug = False

    def toggle_debug (self):
        self.show_debug = not self.show_debug

    def print_state(self):
        if self.show_debug:
            state = self.camera.get_camera_state()
            print("Camera State:", state)
            print("Keys Down:",self.input_state.get_down_keys())
            print("Mouse Position:",self.input_state.mouse_position)
            print("Zoom:", self.camera.zoom)
            print("Velocity:", self.camera.velocity)
            print("")

class CameraShakeEffect:

    def __init__(self, camera: Camera2D):
        self.camera = camera
        self.amplitude = 0.0
        self.duration = 0.0
        self.elapsed = 0.0
        self.enabled = False

    def start_shake(self, amplitude:float, duration:float):
        self.amplitude = amplitude
        self.duration = duration
        self.elapsed = 0.0
        self.enabled = True    

    def update(self, delta_time: float):
        if self.enabled:
            offset_x = random.uniform(-self.amplitude,self.amplitude)
            offset_y = random.uniform(-self.amplitude,self.amplitude)
            self.camera.position[0] += offset_x
            self.camera.position[1] += offset_y
            if self.elapsed >= self.duration:
                self.enabled = False
                self.amplitude = 0.0
                self.duration = 0.0
                self.elapsed = 0.0


class CameraSmoothEasing:

    def __init__(self, camera: Camera2D):
        self.camera = camera

    def ease_in_out(self, start: float, end: float, t:float) -> float:
        t2 = t * t * (3 - 2 * t)
        return start + (end - start) * t2
    
    def interpolate_zoom(self, target_zoom: float, duration: float, time_elapsed: float):
        t = min(time_elapsed / duration, 1.0)
        self.camera.zoom = self.ease_in_out(self.camera.zoom,target_zoom,t)


#This probably has the most no of classes in this whole project.    
class InputGestureRecognizer:
    def __init__(self, input_state: InputState):
        self.input_state = input_state
        self.dragging = False
        self.drag._start_pos = (0, 0)
        self.drag_current_pos = (0, 0)
        self.last_tap_time = 0.0
        self.tap_count = 0

    def update_drag(self):
        if self.input_state.mouse_button_down.get(0, False):
            if not self.dragging:
                self.dragging = True
                self.drag_start_pos = self.input_state.mouse_position
            self.drag_current_pos = self.input_state.mouse_position
        else:
            if self.dragging:
                self.dragging = False
                self.drag_start_pos = (0, 0)
                self.drag_current_pos = (0, 0)

    def detect_tap(self, current_time: float):
        tap_interval = 0.25
        if self.input_state.mouse_buttons_pressed.get(0, False):
            if current_time - self.last_tap_time < tap_interval:
               self.tap_count += 1
            else:   
                self.tap_count = 1
            self.last_tap_time =current_time


    def is_double_tap(self) -> bool:
        return self.tap_count == 2

class CameraBoundaryManager:  
    def __init__(self, camera: Camera2D):     
        self.camera = camera
        self.active = False
        self.margin_x = 50
        self.margin_y = 50

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def enforce(self):
        if not self.active: 
            return
        
        if self.camera.positon[0] < self.camera.bounds_min[0] + self.margin_x:
            self.camera.position[0] = self.camera.bounds_min[0] + self.margin_x

        if self.camera.position[0] > self.camera.bounds_max[0] - self.margin_x:
            self.camera.position[0] = self.camera.bounds_max[0] - self.margin_x

        if self.camera.position[1] < self.camera.bounds_min[1] + self.margin_y:
            self.camera.position[1] = self.camera.bounds_min[1] + self.margin_y

        if self.camera.position[1] > self.camera.bounds_max[1] - self.margin_y:
            self.camera.position[1] = self.camera.bounds_max[1] - self.margin_y


class AdvancedInputHandler:

    def __init__(self, input_state: InputState, camera: Camera2D, input_mapper: InputMapper):
        self.input_state = input_state
        self.camera = camera
        self.input_mapper = input_mapper
        self.last_update_time = time.time()

    def update(self):
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        if self.input_state.is_key_down("left"):
            self.camera.move(-1,0)
        if self.input_state.is_key_down("right"):
            self.camera.move(1,0)
        if self.input_state.is_key_down("up"):
            self.camera.move(0,-1)
        if self.input_state.is_key_down("down"):
            self.camera.move(0,1)
        if self.input_state.is_key_down("zoom_in"):
            self.camera.zoom_in(current_time)
        if self.input_state.is_key_down("zoom_out"):
            self.camera.zoom_out(current_time)
        if self.input_state.is_key_down("focus_center"):
            self.camera.position = [0.0,0.0]
        if self.input_state.is_key_down("reset"):
            self.camera.position = [0.0,0.0]
            self.camera.zoom = 1.0
            self.camera.target_zoom = 1.0
            self.camera.target_zoom = 1.0
        self.last_update_time = current_time
        self.input_state.reset_for_frame()

#Resting for a while