"""Generate denoised GID IMU data with the retained 4- or 6-IMU fusion model."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import OurData
from model import FourHead_ResFusion


# Make every relative checkpoint and data path independent of the launch folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# ---------- Fusion configuration ----------
# Change only this value.  Its checkpoint, SPM prior, and fusion architecture
# are selected together so checkpoint loading remains strict.
IMU_NUM = 6
BATCH_SIZE = 512
SEQ_LEN = 30
FEATURES_PER_IMU = 12  # 9D rotation matrix + 3D acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMU_CONFIGS = {
    4: {
        'fuse_ckpt': 'checkpoint/SDN4/SDN4_fuse.pth',
        'weight_path': './checkpoint/ck.bin',
        'model_kwargs': {'use_cross': True, 'use_late': True},
    },
    6: {
        'fuse_ckpt': 'checkpoint/SDN6/SDN6_fuse.pth',
        'weight_path': 'checkpoint/spm/ses_real_6.pt',
        'model_kwargs': {'use_cross': False, 'use_late': False},
    },
}


def get_imu_config(imu_num: int) -> dict:
    try:
        return IMU_CONFIGS[imu_num]
    except KeyError as error:
        raise ValueError('IMU_NUM must be 4 or 6.') from error


# Default project-local 6-IMU test data.  CLI options can select another GID
# dataset, such as GID_upper with the 4-IMU model.
DATA_ROOT = os.path.join(THIS_DIR, 'GIDData', 'GID_full', 'test')
OUTPUT_TAG = 'denoised'


def imu_loose_to_rot_acc(imu_loose: torch.Tensor, imu_num: int = IMU_NUM):
    """Recover raw rotations and accelerations from ``OurData`` IMU features.

    ``imu_loose`` stores rotations relative to the root and accelerations in
    the root frame, scaled by 1/30.  The final IMU is the root.  This converts
    the denoised features to ``rot_raw [T, IMU, 3, 3]`` and
    ``acc_raw [T, IMU, 3]`` for saving.
    """
    num_frames = imu_loose.shape[0]
    features = imu_loose.reshape(num_frames, imu_num, FEATURES_PER_IMU)
    rot_relative = features[..., :9].reshape(num_frames, imu_num, 3, 3)
    acc_processed = features[..., 9:]

    # R_rel = R_root^T @ R_raw during preprocessing for non-root sensors.
    root_rotation = rot_relative[:, -1]
    sensor_rotation = root_rotation.unsqueeze(1).matmul(rot_relative[:, :-1])
    rot_raw = torch.cat((sensor_rotation, root_rotation.unsqueeze(1)), dim=1)

    # Add root acceleration back to each sensor, then undo the frame rotation
    # and 1/30 scaling used by preprocessing.
    root_acc_processed = acc_processed[:, -1:]
    sensor_acc_processed = acc_processed[:, :-1] + root_acc_processed
    acc_processed = torch.cat((sensor_acc_processed, root_acc_processed), dim=1)
    acc_raw = (acc_processed * 30.0).bmm(root_rotation)
    return rot_raw, acc_raw


def load_fuse_model(imu_num: int) -> FourHead_ResFusion:
    """Build exactly the selected 4- or 6-IMU fusion architecture."""
    config = get_imu_config(imu_num)
    model = FourHead_ResFusion(
        gate_init_ratio_early=0.1,
        gate_init_ratio_cross=0.0,
        gate_init_ratio_late=0.0,
        tok_init_eps=0.1,
        upper_res_scale=0.1,
        in_num_joints=imu_num,
        out_num_joints=imu_num,
        weight_path=config['weight_path'],
        with_ssms=True,
        **config['model_kwargs'],
    ).to(DEVICE)
    checkpoint = torch.load(config['fuse_ckpt'], map_location='cpu')
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def denoise_sequence(
    model: FourHead_ResFusion,
    data_root: str,
    data_dir: str,
    imu_num: int,
    output_tag: str,
):
    """Denoise one ``name_motion`` sequence and save its raw IMU tensors."""
    sequence_dir = os.path.join(data_root, data_dir)
    if not os.path.isdir(sequence_dir):
        print(f'[SKIP] Missing sequence: {sequence_dir}')
        return

    test_data = OurData.load_data(
        data_root,
        data_type=data_dir,
        imu_num=imu_num,
    )
    expected_features = imu_num * FEATURES_PER_IMU
    if test_data['imu_loose'].shape[-1] != expected_features:
        raise ValueError(
            f'Expected {expected_features} input features for {imu_num} IMUs, '
            f'got {test_data["imu_loose"].shape[-1]}.'
        )
    dataset = OurData(
        x=test_data['imu_loose'],
        y=test_data['imu_tight'],
        seq_len=SEQ_LEN,
        step=1,
    )
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    predictions = []
    for inputs, _ in tqdm(loader, desc=data_dir, leave=False):
        predictions.append(model(inputs.to(DEVICE))[:, -1].cpu())

    if not predictions:
        print(f'[SKIP] Sequence is shorter than {SEQ_LEN} frames: {data_dir}')
        return

    # Window ``s:s+SEQ_LEN`` predicts frame ``s+SEQ_LEN-1``.  Keep the first
    # SEQ_LEN-1 raw frames and replace only the frames with predictions.
    predictions = torch.cat(predictions, dim=0)
    denoised_features = test_data['imu_loose'].clone()
    start = SEQ_LEN - 1
    end = start + predictions.shape[0]
    if end > denoised_features.shape[0]:
        raise ValueError(f'Prediction range [{start}:{end}] exceeds {denoised_features.shape[0]} frames.')
    denoised_features[start:end] = predictions

    rot_raw, acc_raw = imu_loose_to_rot_acc(denoised_features, imu_num=imu_num)
    torch.save(rot_raw, os.path.join(sequence_dir, f'rot_{output_tag}.pt'))
    torch.save(
        acc_raw.reshape(-1, imu_num, 3),
        os.path.join(sequence_dir, f'acc_{output_tag}.pt'),
    )
    print(f'[{data_dir}] saved rot {tuple(rot_raw.shape)}, acc {tuple(acc_raw.shape)}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--imu-num', type=int, choices=sorted(IMU_CONFIGS), default=IMU_NUM)
    parser.add_argument('--output-tag', default=OUTPUT_TAG)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'Data root not found: {data_root}')

    model = load_fuse_model(args.imu_num)
    sequence_names = sorted(
        entry.name for entry in os.scandir(data_root) if entry.is_dir()
    )
    for sequence_name in sequence_names:
        denoise_sequence(model, data_root, sequence_name, args.imu_num, args.output_tag)


if __name__ == '__main__':
    main()
