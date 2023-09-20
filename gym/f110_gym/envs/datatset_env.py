import gymnasium as gym
# import so PAth
from pathlib import Path
from f110_gym.envs.f110_env import F110Env
# import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np

import os 
import sys
import pickle
import tensorflow as tf
import zarr
from typing import Union, Tuple, Dict, Optional, List, Any

class F1tenthDatasetEnv(gym.Env):
    def __init__(
        self,
        name,
        f110_gym_kwarg,
        root_dir=None,
        dataset_url=None,
        flatten_obs=True,
        flatten_trajectories = False,
        laser_obs = True,
        subsample_laser = 10, # 1 is no subsampling
        padd_trajectories = True,
        trajectory_max_length = 500,
        as_trajectories = True,
        ):
        if flatten_trajectories:
            raise NotImplementedError("TODO! Implement flatten_trajectories")
        self.name = name
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.padd_trajectories = padd_trajectories
        self.trajectory_max_length = trajectory_max_length
        self.sim_env = F110Env(**f110_gym_kwarg)
        self.flatten_obs = flatten_obs
        self.subsample_laser = subsample_laser
        self.as_trajectories = as_trajectories
        self.laser_obs = laser_obs
        # open with zarr
        # self.dataset = zarr.open(self.root_dir, mode='r')
        # list the subdirs
        # print(self.dataset.tree())


        # now we define our action and observation spaces
        action_space_low = np.array([-1.0,-1])
        action_space_high = np.array([1.0, 1.0])
        self.action_space = gym.spaces.Box(action_space_low, action_space_high)
        
        # observation space is a dict with keys:
        # the first one is the scan
        # the second one is the odometry
        rays = int(1080/subsample_laser)
        # laser_scan_space = gym.spaces.Box(low=np.array([0]*rays), high=np.array([30]*rays))
        # Dict('ang_vels_z': Box(-20.0, 20.0, (1,), float32), 
        # 'linear_vels_x': Box(-20.0, 20.0, (1,), float32), 
        # 'linear_vels_y': Box(-20.0, 20.0, (1,), float32), 
        # 'poses_theta': Box(-1e+30, 1e+30, (1,), float32), 
        # 'poses_x': Box(-30.0, 30.0, (1,), float32), 
        # 'poses_y': Box(-30.0, 30.0, (1,), float32), 
        # 'progress': Box(0.0, 1.0, (1,), float32), 
        # 'lidar_occupancy': Box(0, 255, (1, 80, 80), uint8), 
        # 'previous_action': Box(-1.0, 1.0, (1, 2), float32))
        #state_space = gym.spaces.Box(low=np.array([-100, -100, 0,-10.0, -10.0, -10.0]), high=np.array([100, 100,2*np.pi, 10,10,10]))
        state_space = gym.spaces.Dict({'ang_vels_z': Box(-1.0, 1.0, (1,), np.float32),
        'linear_vels_x': Box(-1.0, 1.0, (1,), np.float32),
        'linear_vels_y': Box(-1.0, 1.0, (1,), np.float32),
        'poses_theta': Box(-10, 10, (1,), np.float32), #TODO!
        'poses_x': Box(-1.0, 1.0, (1,), np.float32),
        'poses_y': Box(-1.0, 1.0, (1,), np.float32),
        'progress': Box(0.0, 1.0, (1,), np.float32),
        # 'lidar_occupancy': ,
        'previous_action': Box(-1.0, 1.0, (1, 2), np.float32)})
        # print(state_space)
        self.observation_space = gym.spaces.Dict({
            # 'laser_scan': Box(0, 255, (1, 80, 80), np.uint8),
            'state': state_space
        })
        self.laser_obs_space = gym.spaces.Box(0, 255, (1, 80, 80), np.uint8)
        # print(self.observation_space)
        self._orig_flat_obs_space = gym.spaces.flatten_space(self.observation_space)
        
        if self.flatten_obs:
            self.observation_space = self._orig_flat_obs_space
        self.dataset = dict(
            actions=[],
            observations=[],
            rewards=[],
            terminals=[],
            infos=[],)

    def shorten_trajectories(self, dataset):
        print("Hi")
        # where are the terminals
        terminals = np.where(dataset['terminals'])[0]
        print(terminals)
        
    def pad_trajectories(self, dataset, trajectory_max_length):
        # split dataset into trajectories at done signal = 1
        trajectories = np.split(dataset['actions'], np.where(dataset['terminals'])[0]+1)[0:-1]
        # padd to max_length
        # for trajectories in 
        print([len(tra) for tra in trajectories])
        print("hi")
        return dataset
    
    def get_change_indices(self, model_names):
        change_indices = []
        current_name = model_names[0]
        for idx, name in enumerate(model_names):
            if name != current_name:
                change_indices.append(idx)
                current_name = name
        return change_indices + [len(model_names)]

    def get_model_names(self, dataset):
        pass
    def get_scan(self, timesteps):
        indices = timesteps - 1
        root = zarr.open(self.zarr_path, mode='r')
        scan = root['observations']['lidar_occupancy'][indices]
        return scan

    def get_dataset(
        self,
        zarr_path: Union[str, os.PathLike] = None,
        clip: bool = True,
        # rng: Optional[Tuple[int, int]] = None,
        without_agents: Optional[np.ndarray] = [],
    ) -> Dict[str, Any]:
        """ 
        TODO! this is copied from https://github.com/rr-learning/trifinger_rl_datasets/blob/master/trifinger_rl_datasets/dataset_env.py
        Get the dataset.

        When called for the first time, the dataset is automatically downloaded and
        saved to ``~/.trifinger_rl_datasets``.

        Args:
            zarr_path:  Optional path to a Zarr directory containing the dataset, which will be
                used instead of the default.
            clip:  If True, observations are clipped to be within the environment's
                observation space.
            rng:  Optional range to return. rng=(m,n) means that observations, actions
                and rewards m to n-1 are returned. If not specified, the entire
                dataset is returned.
            indices: Optional array of timestep indices for which to load data. rng
                and indices are mutually exclusive, only one of them can be set.
            n_threads: Number of threads to use for processing the images. If None,
                the number of threads is set to the number of CPUs available to the
                process.
        Returns:
            A dictionary containing the following keys

                - observations: Either an array or a list of dictionaries
                  containing the observations depending on whether
                  `flatten_obs` is True or False.
                - actions: Array containing the actions.
                - rewards: Array containing the rewards.
                - timeouts: Array containing the timeouts (True only at
                  the end of an episode by default. Always False if
                  `set_terminals` is True).
                - terminals: Array containing the terminals (Always
                  False by default. If `set_terminals` is True, only
                  True at the last timestep of an episode).
                - images (only if present in dataset): Array of the
                  shape (n_control_timesteps, n_cameras, n_channels,
                  height, width) containing the image data. The cannels
                  are ordered as RGB.
        """
        # The ORL Dataset is loaded from Zarr arrays that look as follows:
        # TODO!
        if zarr_path is None:
            raise NotImplementedError("TODO! Download the dataset from the web")
            # zarr_path = self._download_dataset()
        
        #store = zarr.LMDBStore(zarr_path, readonly=True)
        print(zarr_path)
        self.zarr_path = zarr_path
        root = zarr.open(zarr_path, mode='r') #(store=store)

        #print(root.tree())
        model_names = root['model_name'][:]
        # print all unique model names
        print(np.unique(model_names))
        data_dict = {}
        indices = np.where(~np.isin(model_names, without_agents))[0]
        # print("Indices:", indices)
        data_dict['rewards'] = root['rewards'][indices]
        # print("hi")
        data_dict['terminals'] = np.logical_or(root['done'][indices],root['truncated'][indices])
        # print("hi")
        data_dict['actions'] = root['actions'][indices]
        # print("hi")
        data_dict['timestep'] = root['timestep'][indices]
        # loop over observation keys
        data_dict['observations'] = dict()
        for key in root['observations'].array_keys():
            # print(key)
            if key != 'lidar_occupancy':
                data_dict['observations'][key] = root['observations'][key][indices]
            else:
                continue
                # data_dict['scan'] = root['observations'][key][indices]
        # data_dict = root['observations'][indices]
        # New code for flattening
        if self.flatten_obs:
            # Extract all the filtered arrays and flatten them
            # Expand dimensions and prepare arrays for concatenation
            # print([arr.shape for arr in data_dict['observations'][key]])
            arrays_to_concat = [data_dict['observations'][key].reshape([data_dict['observations'][key].shape[0], -1]) for key in data_dict['observations']]
    
            # print the shapes of all arrays to concatenate
            # print([arr.shape for arr in arrays_to_concat])
            # Concatenate all arrays along the last dimension
            concatenated_obs = np.concatenate(arrays_to_concat, axis=-1)
            
            # Reshape the concatenated array: [timesteps, -1] means keeping the number of timesteps the same, but flattening all other dimensions.
            data_dict['observations'] = concatenated_obs.reshape([concatenated_obs.shape[0], -1])
            
        data_dict['infos'] = dict()
        # print("hi")
        data_dict['infos']['model_name']= root['model_name'][indices]
        return data_dict