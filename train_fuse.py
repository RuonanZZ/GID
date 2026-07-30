"""Train the GID fusion denoiser with either four or six IMUs."""

import os

import torch
import torch.nn as nn

from Aplus.runner import BaseEvaluator
from data import OurData
from model import FourHead_ResFusion
from runner import MyTrainer


# Set this to 4 or 6.  The matching IMU order, SPM prior, heads, and output
# checkpoint directory are selected together.
IMU_NUM = 6
BATCH_SIZE = 512
SEQ_LEN = 30
LR = 5e-4
EPOCHS = 20
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GIDData', 'GID_full')


IMU_CONFIGS = {
    4: {
        'names': ('left', 'right', 'back', 'root'),
        'weight_path': './checkpoint/ck.bin',
        'run_name': 'fuse_early',
        'fusion_kwargs': {'use_cross': True, 'use_late': True},
        'ckpt_map': {
            'left': 'checkpoint/SDN4/SDN4_left.pth',
            'right': 'checkpoint/SDN4/SDN4_right.pth',
            'back': 'checkpoint/SDN4/SDN4_back.pth',
            'root': 'checkpoint/SDN4/SDN4_root.pth',
        },
    },
    6: {
        'names': ('left', 'right', 'l_leg', 'r_leg', 'back', 'root'),
        'weight_path': 'checkpoint/spm/ses_real_6.pt',
        'run_name': 'fuse6',
        'fusion_kwargs': {'use_cross': False, 'use_late': False},
        'ckpt_map': {
            'left': 'checkpoint/SDN6/SDN6_left.pth',
            'right': 'checkpoint/SDN6/SDN6_right.pth',
            'l_leg': 'checkpoint/SDN6/SDN6_l_leg.pth',
            'r_leg': 'checkpoint/SDN6/SDN6_r_leg.pth',
            'back': 'checkpoint/SDN6/SDN6_back.pth',
            'root': 'checkpoint/SDN6/SDN6_root.pth',
        },
    },
}


def get_imu_config(imu_num: int) -> dict:
    try:
        return IMU_CONFIGS[imu_num]
    except KeyError as error:
        raise ValueError('IMU_NUM must be 4 or 6.') from error


def build_fusion_model(imu_num: int, device: torch.device) -> FourHead_ResFusion:
    config = get_imu_config(imu_num)
    model = FourHead_ResFusion(
        gate_init_ratio_early=0.1,
        gate_init_ratio_cross=0.0,
        gate_init_ratio_late=0.0,
        tok_init_eps=0.1,
        in_num_joints=imu_num,
        out_num_joints=imu_num,
        weight_path=config['weight_path'],
        with_ssms=True,
        **config['fusion_kwargs'],
    ).to(device)
    model.load_heads(config['ckpt_map'], order=config['names'], strict=True, freeze=True)
    return model


def main():
    config = get_imu_config(IMU_NUM)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Use the project-local, anonymized GID train/test split.
    train_raw = OurData.load_data(
        folder_path=os.path.join(DATA_ROOT, 'train'), imu_num=IMU_NUM,
    )
    test_raw = OurData.load_data(
        folder_path=os.path.join(DATA_ROOT, 'test'), imu_num=IMU_NUM,
    )
    print(train_raw['imu_loose'].shape, test_raw['imu_tight'].shape)

    data_train = OurData(
        x=train_raw['imu_loose'], y=train_raw['imu_tight'], seq_len=SEQ_LEN, step=1,
    )
    data_eval = OurData(
        x=test_raw['imu_loose'], y=test_raw['imu_tight'], seq_len=SEQ_LEN, step=1,
    )

    model = build_fusion_model(IMU_NUM, device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LR,
        weight_decay=LR * 1e-1,
    )
    criterion = nn.MSELoss()
    trainer = MyTrainer(
        model=model, data=data_train, optimizer=optimizer,
        batch_size=BATCH_SIZE, loss_func=criterion,
    )
    evaluator = BaseEvaluator.from_trainner(trainer, data_eval=data_eval, loss_func=criterion)

    for _ in range(EPOCHS):
        trainer.run(
            epoch=1, evaluator=evaluator,
            data_shuffle=True, verbose=False, batch_sampler=None,
        )
        trainer.save(folder_path=f"checkpoint/{config['run_name']}", model_name=config['run_name'])
        trainer.log_export(f"log/{config['run_name']}.xlsx")


if __name__ == '__main__':
    main()
