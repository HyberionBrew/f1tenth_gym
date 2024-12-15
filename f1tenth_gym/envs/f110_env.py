# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Hongrui Zheng
"""

# gym imports
import gymnasium as gym

from .action import (CarAction,
                                  from_single_to_multi_action_space)
from .integrator import IntegratorType
from .rendering import make_renderer

from .track import Track

# base classes
from .base_classes import Simulator, DynamicModel
from .observation import observation_factory
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update


# others
import numpy as np


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment.

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """

    # NOTE: change matadata with default rendering-modes, add definition of render_fps
    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 100}

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        super().__init__()

        # Configuration
        print("In gym config:", config)
        self.config = self.default_config()
        self.configure(config)

        self.seed = self.config["seed"]
        self.map = self.config["map"]
        self.params = self.config["params"]
        self.num_agents = self.config["num_agents"]
        self.timestep = self.config["timestep"]
        self.ego_idx = self.config["ego_idx"]
        self.integrator = IntegratorType.from_string(self.config["integrator"])
        self.model = DynamicModel.from_string(self.config["model"])
        self.observation_config = self.config["observation_config"]
        self.action_type = CarAction(self.config["control_input"], params=self.params)

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            model=self.model,
            action_type=self.action_type,
        )
        self.sim.set_map(self.map)

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(
                self.map
            )  # load track in gym env for convenience

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        assert (
            "type" in self.observation_config
        ), "observation_config must contain 'type' key"
        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()

        # action space
        self.action_space = from_single_to_multi_action_space(
            self.action_type.space, self.num_agents
        )

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.render_obs = None
        self.render_mode = render_mode

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )
        
        ## added by me to do crash detection easily
        # initalize a scan simulator 
        from .laser_models import ScanSimulator2D
        # make the scan go around the car
        self.scan_simulator = ScanSimulator2D(64, 6.28) 
        self.scan_simulator.set_map(self.track)
        self.add_render_callback(self.render_monitor)
        self.aux_data = None
    
    def render_monitor_old(self,e):
        if self.aux_data is not None:
            #print(self.aux_data[0])
            #print(self.aux_data[1])
            points = np.asarray(self.aux_data["data"])
            #print(points.shape)
            safe =self.aux_data["violation"]
            #print(safe)
            last_violation = self.aux_data["last_violation"]
            # make the points green if safe
            if last_violation != 0:
                e.render_points(points, size=1, color=(255, 0, 0))
            else:
                e.render_points(points, size=1, color=(0, 255, 0))
    def render_monitor(self, e):
        """
        Renders monitoring points with appropriate colors based on agent roles and violations.

        Args:
            e: The environment or rendering engine instance with a `render_points` method.
        """
        if self.aux_data is not None:
            points = np.asarray(self.aux_data["data"])  # Shape: (num_agents * points_per_agent, 2)
            #print(points)
            num_agents = getattr(self, 'num_agents', 1)  # Default to 1 if not defined
            # check if points is empty
            if len(points) == 0 or len(points[0]) == 0:  
                return
                
            if num_agents > 1:
                # take first two dimension
                points = points[:, :2]
                
                points_per_agent = len(points) // num_agents
                
                # Ensure that points can be evenly divided among agents
                if len(points) % num_agents != 0:
                    raise ValueError("The number of points is not divisible by the number of agents.")

                # Reshape points to (num_agents, points_per_agent, 2)
                agent_points = points.reshape((num_agents, points_per_agent, 2))

                # Determine color for ego agent based on violation
                if self.aux_data.get("last_violation", 0) != 0:
                    ego_color = (255, 0, 0)  # Red
                else:
                    ego_color = (0, 255, 0)  # Green

                # Color for other agents
                other_color = (255, 165, 0)  # Orange

                # Prepare a list of colors: first agent's color followed by others
                colors = [ego_color] + [other_color] * (num_agents - 1)

                # Convert colors to a NumPy array for efficient processing
                colors_np = np.array(colors, dtype=np.uint8)  # Shape: (num_agents, 3)

                # Iterate over agents and render their points with respective colors
                for agent_idx in range(num_agents):
                    e.render_points(agent_points[agent_idx], size=1, color=colors_np[agent_idx].tolist())
            else:
                # Single agent: original behavior
                
                #print(points)
                #print(len(points))
                # take the first two dimensions
                points = points[:, :2]
                last_violation = self.aux_data.get("last_violation", 0)
                if last_violation != 0:
                    color = (255, 0, 0)  # Red
                else:
                    color = (0, 255, 0)  # Green
                
                e.render_points(points, size=1, color=color)

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().

        Args:
            None

        Returns:
            a configuration dict
        """
        return {
            "seed": 12345,
            "map": "Spielberg",
            "params": {
                "mu": 1.0489,
                "C_Sf": 4.718,
                "C_Sr": 5.4562,
                "lf": 0.15875,
                "lr": 0.17145,
                "h": 0.074,
                "m": 3.74,
                "I": 0.04712,
                "s_min": -0.4189,
                "s_max": 0.4189,
                "sv_min": -3.2,
                "sv_max": 3.2,
                "v_switch": 7.319,
                "a_max": 9.51,
                "v_min": -5.0,
                "v_max": 20.0,
                "width": 0.31,
                "length": 0.58,
            },
            "num_agents": 2,
            "timestep": 0.01,
            "ego_idx": 0,
            "integrator": "rk4",
            "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": None},
            "reset_config": {"type": None},
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config = deep_update(self.config, config)
            self.params = self.config["params"]

            if hasattr(self, "sim"):
                self.sim.update_params(self.config["params"])

            if hasattr(self, "action_space"):
                # if some parameters changed, recompute action space
                self.action_type = CarAction(self.config["control_input"], params=self.params)
                self.action_space = from_single_to_multi_action_space(
                    self.action_type.space, self.num_agents
                )
    def obs_to_monitor(self, obs):
        # return the first poses x the first poses_y linear_vels_x linear_vels_y ang_vels_z
        if self.num_agents == 1:
            trimmed_obs = np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], obs["linear_vels_x"][0], obs["linear_vels_y"][0], obs["ang_vels_z"][0]])
        else:
            # append all the agents
            all_obs = []
            for i in range(self.num_agents):
                all_obs.append([obs["poses_x"][i], obs["poses_y"][i], obs["poses_theta"][i], obs["linear_vels_x"][i], obs["linear_vels_y"][i], obs["ang_vels_z"][i]])
            trimmed_obs = np.array(all_obs)
        return trimmed_obs

    def is_safe(self, obs, dang_dist = 0.3, **kwargs):
        """ Checks if we are crashing at the x,y position provided.
        Args:
            obs (_type_): Assumes that the first value is the x and second the y
        """
        #print(obs[0:3])
        #print(np.asarray([obs[0],obs[1],0.0]))
        scan = self.scan_simulator.scan(np.asarray([obs[0],obs[1],0.0]),None)
        # check if each scan element is less than the dang_dist
        is_safe =  min(scan) > dang_dist
        #print(is_safe, dang_dist, min(scan))
        return is_safe
    
    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    def _update_state(self):
        """
        Update the env's states according to observations.
        """
        self.poses_x = self.sim.agent_poses[:, 0]
        self.poses_y = self.sim.agent_poses[:, 1]
        self.poses_theta = self.sim.agent_poses[:, 2]
        self.collisions = self.sim.collisions

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)

        # observation
        obs = self.observation_type.observe()

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "steering_angles": self.sim.agent_steerings,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
        }
        #print("collisons", self.sim.collisions)

        # check done
        done, toggle_list = self._check_done()
        truncated = False
        info = {"checkpoint_done": toggle_list}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        # print("reset poses", poses)
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        return obs, info

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        self.sim.set_map(map_name)
        self.track = Track.from_track_name(map_name)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()
