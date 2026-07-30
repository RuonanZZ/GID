"""Train one GID denoiser per IMU for the four- or six-IMU setup."""

import os

import torch
import torch.nn as nn

from Aplus.runner import BaseEvaluator
from Common.models import Inertial_PoseTransformer_Upper
from data import OurData
from runner import MyTrainer


# Set this to 4 or 6.  The IMU order, learning rates, SPM prior, and output
# checkpoint prefix are selected together.
IMU_NUM = 6
BATCH_SIZE = 128
SEQ_LEN = 30
EPOCHS = 10
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GIDData', 'GID_full')


IMU_CONFIGS = {
    4: {
        'names': ('left', 'right', 'back', 'root'),
        'learning_rates': (5e-4, 5e-4, 1e-4, 1e-4),
        'weight_path': './checkpoint/ck.bin',
        'checkpoint_prefix': 'SDN4',
        'checkpoint_dir': 'checkpoint/SDN4',
    },
    6: {
        'names': ('left', 'right', 'l_leg', 'r_leg', 'back', 'root'),
        'learning_rates': (2e-4, 2e-4, 5e-4, 5e-4, 2e-6, 1e-4),
        'weight_path': 'checkpoint/spm/ses_real_6.pt',
        'checkpoint_prefix': 'SDN6',
        'checkpoint_dir': 'checkpoint/SDN6',
    },
}


def get_imu_config(imu_num: int) -> dict:
    try:
        return IMU_CONFIGS[imu_num]
    except KeyError as error:
        raise ValueError('IMU_NUM must be 4 or 6.') from error


def main():
    config = get_imu_config(IMU_NUM)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Use the same project-local, anonymized GID split as train_fuse.py.
    train_raw = OurData.load_data(
        folder_path=os.path.join(DATA_ROOT, 'train'), imu_num=IMU_NUM,
    )
    test_raw = OurData.load_data(
        folder_path=os.path.join(DATA_ROOT, 'test'), imu_num=IMU_NUM,
    )
    print(train_raw['imu_loose'].shape, test_raw['imu_tight'].shape)

    for imu_id, (imu_name, learning_rate) in enumerate(
        zip(config['names'], config['learning_rates'])
    ):
        data_train = OurData(
            x=train_raw['imu_loose'],
            y=train_raw['imu_tight'][:, imu_id * 12:(imu_id + 1) * 12],
            seq_len=SEQ_LEN,
            step=1,
            shuffle=True,
        )
        data_eval = OurData(
            x=test_raw['imu_loose'],
            y=test_raw['imu_tight'][:, imu_id * 12:(imu_id + 1) * 12],
            seq_len=SEQ_LEN,
            step=1,
        )

        model = Inertial_PoseTransformer_Upper(
            in_num_joints=IMU_NUM,
            in_chans=12,
            num_frame=SEQ_LEN,
            out_num_joints=1,
            out_chans=12,
            with_spatial_block=True,
            with_spatial_pos_embed=True,
            with_temporal_pos_embed=True,
            with_ssms=True,
            with_ssmt=True,
            weight_path=config['weight_path'],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), learning_rate, weight_decay=learning_rate * 1e-1,
        )
        trainer = MyTrainer(
            model=model,
            data=data_train,
            optimizer=optimizer,
            batch_size=BATCH_SIZE,
            loss_func=nn.MSELoss(),
        )
        evaluator = BaseEvaluator.from_trainner(trainer, data_eval=data_eval, loss_func=nn.MSELoss())
        model_name = f"{config['checkpoint_prefix']}_{imu_name}"

        for _ in range(EPOCHS):
            trainer.run(epoch=1, evaluator=evaluator, data_shuffle=True, verbose=False)
            trainer.save(folder_path=config['checkpoint_dir'], model_name=model_name)
            trainer.log_export(f'log/{model_name}.xlsx')


if __name__ == '__main__':
    main()
