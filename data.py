"""GID IMU loading and fixed-length window utilities."""

import os

import torch
from torch.utils.data import Dataset

from Aplus.data.dataset import random_index


class OurData(Dataset):
    """Create windows and load the current 4- or 6-IMU GID feature format."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None,
                 seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step] if y2 is not None else None
        self.seq_len = seq_len
        self.data_len = max(0, len(self.x) - seq_len)
        self.indexer = (
            random_index(data_len=self.data_len, seed=42)
            if shuffle else list(range(self.data_len))
        )

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        start = self.indexer[index % self.data_len]
        data = [
            self.x[start:start + self.seq_len],
            self.y[start:start + self.seq_len],
        ]
        if self.y2 is not None:
            data.append(self.y2[start:start + self.seq_len])
        return tuple(data)

    @staticmethod
    def load_data(folder_path: str, data_type='all', imu_num=4) -> dict:
        """Load loose/tight IMU recordings in GID's fixed 9D-rotation layout."""
        loose_rotations, loose_accelerations = [], []
        tight_rotations, tight_accelerations = [], []

        for root, dirs, _ in os.walk(folder_path):
            for dir_name in dirs:
                if data_type != 'all' and data_type not in dir_name:
                    continue

                sequence_dir = os.path.join(root, dir_name)
                suffix = f'_{imu_num}' if os.path.exists(
                    os.path.join(sequence_dir, f'rot_{imu_num}.pt')
                ) else ''
                loose_rotation = torch.load(
                    os.path.join(sequence_dir, f'rot{suffix}.pt')
                )
                loose_acceleration = torch.load(
                    os.path.join(sequence_dir, f'acc{suffix}.pt')
                ).reshape(-1, imu_num, 3)
                loose_rotations.append(loose_rotation)
                loose_accelerations.append(loose_acceleration)

                tight_rotation_path = os.path.join(sequence_dir, f'vrot{suffix}.pt')
                if os.path.exists(tight_rotation_path):
                    tight_rotations.append(torch.load(tight_rotation_path))
                    tight_accelerations.append(torch.load(
                        os.path.join(sequence_dir, f'vacc{suffix}.pt')
                    ).reshape(-1, imu_num, 3))
                else:
                    tight_rotations.append(torch.zeros_like(loose_rotation))
                    tight_accelerations.append(torch.zeros_like(loose_acceleration))

        if not loose_rotations:
            raise FileNotFoundError(
                f'No matching IMU recordings in {folder_path!r} for {data_type!r}.'
            )

        loose_rotation = torch.cat(loose_rotations, dim=0)
        loose_acceleration = torch.cat(loose_accelerations, dim=0).reshape(-1, imu_num, 3)
        tight_rotation = torch.cat(tight_rotations, dim=0)
        tight_acceleration = torch.cat(tight_accelerations, dim=0).reshape(-1, imu_num, 3)
        imu_loose, imu_tight = OurData.build_imu_features(
            loose_rotation, loose_acceleration, tight_rotation, tight_acceleration,
        )
        return {'imu_loose': imu_loose, 'imu_tight': imu_tight}

    @staticmethod
    def build_imu_features(loose_rotation, loose_acceleration,
                           tight_rotation, tight_acceleration):
        """Convert raw IMU tensors to the shared model/evaluation feature format."""
        loose_acceleration, tight_acceleration = OurData._normalize_acceleration(
            loose_acceleration, tight_acceleration, loose_rotation, tight_rotation,
        )
        loose_rotation, tight_rotation = OurData._relative_rotations(
            loose_rotation, tight_rotation,
        )
        return (
            torch.cat((loose_rotation, loose_acceleration), dim=-1).flatten(1),
            torch.cat((tight_rotation, tight_acceleration), dim=-1).flatten(1),
        )

    @staticmethod
    def _normalize_acceleration(loose_acceleration, tight_acceleration,
                                loose_rotation, tight_rotation):
        loose_acceleration = torch.clamp(loose_acceleration, min=-60, max=60)
        tight_acceleration = torch.clamp(tight_acceleration, min=-60, max=60)
        loose_acceleration = torch.cat((
            loose_acceleration[:, :-1] - loose_acceleration[:, -1:],
            loose_acceleration[:, -1:],
        ), dim=1).bmm(loose_rotation[:, -1].transpose(1, 2)) / 30
        tight_acceleration = torch.cat((
            tight_acceleration[:, :-1] - tight_acceleration[:, -1:],
            tight_acceleration[:, -1:],
        ), dim=1).bmm(tight_rotation[:, -1].transpose(1, 2)) / 30
        return loose_acceleration, tight_acceleration

    @staticmethod
    def _relative_rotations(loose_rotation, tight_rotation):
        loose_rotation = torch.cat((
            loose_rotation[:, -1:].transpose(2, 3).matmul(loose_rotation[:, :-1]),
            loose_rotation[:, -1:],
        ), dim=1)
        tight_rotation = torch.cat((
            tight_rotation[:, -1:].transpose(2, 3).matmul(tight_rotation[:, :-1]),
            tight_rotation[:, -1:],
        ), dim=1)
        return loose_rotation.flatten(2), tight_rotation.flatten(2)
