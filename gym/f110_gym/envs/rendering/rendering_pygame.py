import warnings

import numpy as np
import pygame
from pygame.locals import *

from f110_gym.envs import get_vertices
from f110_gym.envs.rendering.renderer import EnvRenderer, RenderSpec


class PygameEnvRenderer(EnvRenderer):

    def __init__(self, render_spec: RenderSpec):
        self.window = None
        self.canvas = None
        self.clock = None
        self.render_fps = render_spec.render_fps
        self.render_mode = render_spec.render_mode

        width, height = 1600, 1600  # render_spec.window_width, render_spec.window_height
        self.zoom_level = render_spec.zoom_in_factor

        self.car_length = render_spec.car_length
        self.car_width = render_spec.car_width

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.event.set_allowed([])
            flags = DOUBLEBUF
            self.window = pygame.display.set_mode((width, height), flags, 16)
            self.clock = pygame.time.Clock()

        self.poses = None
        self.colors = None
        self.track_map = None

    def update(self, obs):
        self.colors = [(255, 0, 0) if obs['collisions'][i] else (0, 125, 0) for i in range(len(obs["poses_x"]))]
        self.poses = np.stack((obs['poses_x'], obs['poses_y'], obs['poses_theta'])).T

    def add_render_callback(self, callback_fn: callable):
        warnings.warn("add_render_callback is not implemented for PygameEnvRenderer")

    def render_map(self):
        if self.track_map is None:
            track_map = self.map_img  # shape (W, H)
            track_map = np.stack([track_map, track_map, track_map], axis=-1)  # shape (W, H, 3)
            track_map = np.rot90(track_map, k=1)  # rotate clockwise
            track_map = np.flip(track_map, axis=0)  # flip vertically
            self.track_map = track_map
        self.canvas = pygame.surfarray.make_surface(self.track_map)

    def render(self):
        self.canvas.fill((0, 0, 0))  # fill canvas with black
        self.render_map()

        origin = self.map_metadata['origin']
        resolution = self.map_metadata['resolution']
        for i in range(len(self.poses)):
            color, pose = self.colors[i], self.poses[i]

            vertices = get_vertices(pose, self.car_length, self.car_width)
            vertices[:, 0] = (vertices[:, 0] - origin[0]) / resolution
            vertices[:, 1] = (vertices[:, 1] - origin[1]) / resolution

            pygame.draw.lines(self.canvas, color, True, vertices, 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))
