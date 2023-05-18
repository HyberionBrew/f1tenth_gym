import pathlib
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image


@dataclass
class RenderSpec:
    window_width: int
    window_height: int
    zoom_in_factor: float
    render_mode: str
    render_fps: int

    car_length = float
    car_width = float

    def __init__(self, window_width=1000, window_height=800, zoom_in_factor=1.2, render_fps=30,
                 car_length=0.58, car_width=0.31, render_mode="human"):
        self.window_width = window_width
        self.window_height = window_height
        self.zoom_in_factor = zoom_in_factor
        self.render_fps = render_fps

        self.car_length = car_length
        self.car_width = car_width

        self.render_mode = render_mode


class EnvRenderer:
    def load_map(self, map_filepath: pathlib.Path):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (pathlib.Path): path to the map image file

        Returns:
            None
        """

        # load map metadata
        map_yaml = map_filepath.with_suffix('.yaml')
        with open(map_yaml, 'r') as yaml_stream:
            try:
                self.map_metadata = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        self.map_img = np.array(Image.open(map_filepath).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)

    @abstractmethod
    def update(self, state):
        """
        Update the state to be rendered.
        This is called at every rendering call.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_render_callback(self, callback_fn: callable):
        """
        Add a callback function to be called at every rendering call.
        This is called at the end of `update`.
        """
        raise NotImplementedError()

    @abstractmethod
    def render_map(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()
