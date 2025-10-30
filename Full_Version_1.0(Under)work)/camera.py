import time
from typing import Dict, List

class InputState:
    def __init__(self):
        self.keys_down: Dict[str, bool] = {}
        self.keys_pressed: Dict[str, bool] = {}
        self.keys_released: Dict[str, bool] = {}
        self.mouse_buttons_down: Dict[int, bool] = {}
        self.mouse_buttons_pressed: Dict[int, bool] = {}
        self.mouse_buttons_released: Dict[int, bool] = {}
        self.mouse_position = (0, 0)
        self.mouse_delta = (0, 0)
        self.mouse_wheel_delta = 0
        self.last_mouse_position = (0, 0)
        self.double_click_time = 0.3
        self.last_click_time = 0
        self.click_positions: List[tuple] = []
        self.key_repeat_delay = 0.5
        self.key_repeat_rate = 0.05
        self.key_press_times: Dict[str, float] = {}
        self.modifiers = {"shift": False, "ctrl": False, "alt": False}

    def update_modifiers(self, keys: List[str]):
        self.modifiers["shift"] = "shift" in keys
        self.modifiers["ctrl"] = "ctrl" in keys
        self.modifiers["alt"] = "alt" in keys

    def is_double_clicked(self, button: int, current_time: float):
        return (current_time - self.last_click_time) < self.double_click_time \
            and self.mouse_buttons_pressed.get(button, False)


    def key_down(self, key: str, current_time: float):
        if not self.keys_down.get(key, False):
            self.keys_pressed[key] = True
            self.key_press_times[key] = current_time
        self.keys_down[key] = True

    def key_up(self, key: str):
        self.keys_down[key] = False
        self.keys_released[key] = True
        self.keys_pressed[key] = False
        self.key_press_times.pop(key, None)

    def is_key_pressed(self, key: str) -> bool:
        return self.keys_pressed.get(key, False)

    def is_key_down(self, key: str) -> bool:
        return self.keys_down.get(key, False)

    def key_repeat(self, key: str, current_time: float) -> bool:
        if self.is_key_down(key):
            pressed_time = self.key_press_times.get(key, 0)
            elapsed = current_time - pressed_time
            if elapsed > self.key_repeat_delay:
                repeats = (elapsed - self.key_repeat_delay) / self.key_repeat_rate
                return repeats >= 1
        return False

    def mouse_button_down(self, button: int, position: tuple, current_time: float):
        if not self.mouse_buttons_down.get(button, False):
            self.mouse_buttons_pressed[button] = True
            self.last_click_time = current_time
            self.click_positions.append(position)
        self.mouse_buttons_down[button] = True

    def mouse_button_up(self, button: int):
        self.mouse_buttons_down[button] = False
        self.mouse_buttons_released[button] = True
        self.mouse_buttons_pressed[button] = False

    def update_mouse_position(self, position: tuple):
        self.mouse_delta = (position[0] - self.mouse_position[0], position[1] - self.mouse_position[1])
        self.last_mouse_position = self.mouse_position
        self.mouse_position = position

    def update_mouse_wheel(self, delta: int):
        self.mouse_wheel_delta = delta

    def reset_for_frame(self):
        self.keys_pressed.clear()
        self.keys_released.clear()
        self.mouse_buttons_pressed.clear()
        self.mouse_buttons_released.clear()
        self.mouse_wheel_delta = 0
        self.mouse_delta = (0, 0)
        self.click_positions.clear()

    def get_pressed_keys(self) -> List[str]:
        return [k for k, v in self.keys_pressed.items() if v]

    def get_down_keys(self) -> List[str]:
        return [k for k, v in self.keys_down.items() if v]

    def get_released_keys(self) -> List[str]:
        return [k for k, v in self.keys_released.items() if v]

    def get_pressed_mouse_buttons(self) -> List[int]:
        return [b for b, v in self.mouse_buttons_pressed.items() if v]

    def get_down_mouse_buttons(self) -> List[int]:
        return [b for b, v in self.mouse_buttons_down.items() if v]

    def get_released_mouse_buttons(self) -> List[int]:
        return [b for b, v in self.mouse_buttons_released.items() if v]


class Camera2D:
    def __init__(self, viewport_width: int, viewport_height: int):
        self.position = [0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.move_speed = 500.0
        self.zoom_speed = 1.1
        self.smooth_factor = 0.2
        self.bounds_enabled = False
        self.bounds_min = [float('-inf'), float('-inf')]
        self.bounds_max = [float('inf'), float('inf')]
        self.inertia_friction = 0.9
        self.edge_scroll_active = True
        self.edge_scroll_size = 20
        self.edge_scroll_speed = 300.0
        self.wrapping_enabled = False
        self.wrap_bounds_x = (0.0, 1000.0)
        self.wrap_bounds_y = (0.0, 1000.0)
        self.last_zoom_input_time = 0.0
        self.zoom_cooldown = 0.1

    def set_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float):
        self.bounds_enabled = True
        self.bounds_min = [min_x, min_y]
        self.bounds_max = [max_x, max_y]

    def enable_wrapping(self, min_x: float, min_y: float, max_x: float, max_y: float):
        self.wrapping_enabled = True
        self.wrap_bounds_x = (min_x, max_x)
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
                self.position[0] -= width
            if self.position[1] < self.wrap_bounds_y[0]:
                self.position[1] += height
            elif self.position[1] > self.wrap_bounds_y[1]:
                self.position[1] -= height

    def move(self, delta_x: float, delta_y: float):
        self.velocity[0] += delta_x * self.move_speed
        self.velocity[1] += delta_y * self.move_speed

    def zoom_in(self, current_time: float):
        if current_time - self.last_zoom_input_time > self.zoom_cooldown:
            self.target_zoom = min(self.target_zoom * self.zoom_speed, self.max_zoom)
            self.last_zoom_input_time = current_time

    def zoom_out(self, current_time: float):
        if current_time - self.last_zoom_input_time > self.zoom_cooldown:
            self.target_zoom = max(self.target_zoom / self.zoom_speed, self.min_zoom)
            self.last_zoom_input_time = current_time

    def update(self, delta_time: float):
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
        return (world_x, world_y)

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
        if mouse_y < self.edge_scroll_size:
            scroll_y = -self.edge_scroll_speed * (1 - mouse_y / self.edge_scroll_size)
        elif mouse_y > self.viewport_height - self.edge_scroll_size:
            scroll_y = self.edge_scroll_speed * (1 - (self.viewport_height - mouse_y) / self.edge_scroll_size)
        self.move(scroll_x * delta_time, scroll_y * delta_time)

    def get_camera_state(self) -> dict:
        return {
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "zoom": self.zoom,
            "target_zoom": self.target_zoom,
            "bounds_enabled": self.bounds_enabled,
            "bounds_min": tuple(self.bounds_min),
            "bounds_max": tuple(self.bounds_max),
            "wrapping_enabled": self.wrapping_enabled,
            "wrap_bounds_x": self.wrap_bounds_x,
            "wrap_bounds_y": self.wrap_bounds_y
        }
    

    def ease_to_position(self, target, duration, elapsed=0.0):
        if elapsed < duration:
            t = elapsed / duration
            self.position[0] += (target[0] - self.position[0]) * t
            self.position[1] += (target[1] - self.position[1]) * t
        else:
            self.position = list(target)

    def shake(self, amplitude, duration):
        import random
        for _ in range(int(duration * 60)):  # assume 60 FPS
            self.position[0] += random.uniform(-amplitude, amplitude)
            self.position[1] += random.uniform(-amplitude, amplitude)

    def save_state(self):
        return {
            "position": tuple(self.position),
            "zoom": self.zoom,
            "velocity": tuple(self.velocity)
        }

    def load_state(self, state):
        self.position = list(state["position"])
        self.zoom = state["zoom"]
        self.velocity = list(state["velocity"])

    def focus_on(self, x, y, smooth=True):
        if smooth:
            self.target_pos = [x, y]

        else:
            self.position = [x, y]
