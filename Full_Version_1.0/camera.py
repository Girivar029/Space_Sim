import math
from typing import Dict, List

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