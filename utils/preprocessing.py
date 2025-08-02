import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

class GaussianNoiseTransform:
    def __init__(self, mean=0.0, stddev=0.01):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, point_cloud):
        noise = np.random.normal(loc=self.mean, scale=self.stddev, size=point_cloud.shape)
        return point_cloud + noise


class RandomRotationTransform:
    def __init__(self, axis='z'):
        self.axis = axis

    def __call__(self, point_cloud):
        angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians

        if self.axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])

        elif self.axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

        else:  
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])

        if point_cloud.shape[-1] == 4:
            points = point_cloud[:, :3]
            intensity = point_cloud[:, 3]

            points = np.dot(points, rotation_matrix.T)

            rotated_points = np.hstack((points, intensity.reshape(-1, 1)))
        else:
            rotated_points = np.dot(point_cloud, rotation_matrix.T)

        return rotated_points



class FlorTransform:
    def __init__(self, prob:float=0.5, z_th:float=0.5):
        self.z_th = z_th
        self.prob = prob
        self.rng = np.random.default_rng()

    def __call__(self, point_cloud):
        
        if self.rng.random()<=self.prob:
            idx_point = np.where(point_cloud[:, 2]>=self.z_th)[0]
            return point_cloud[idx_point]
        else:
            return point_cloud

class PILNumpyToTensor:

    def __call__(self, x):
        if isinstance(x, Image.Image):
            return pil_to_tensor(x).float()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        else:
            raise TypeError("Input should be numpy of PIL.Image.")
    

