"""Evaluate raw and saved denoised IMU data with ``test_all.py``'s MAE rule."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data import OurData


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'GIDData' / 'GID_full' / 'test'
SEQ_LEN = 30
FEATURES_PER_IMU = 12

IMU_CONFIGS = {
    4: ('left', 'right', 'back', 'root'),
    6: ('left', 'right', 'l_leg', 'r_leg', 'back', 'root'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--imu-num', type=int, choices=sorted(IMU_CONFIGS), default=6)
    parser.add_argument(
        '--denoise-tag', default='denoised',
        help=(
            'Filename tag after the IMU suffix. For example, ``denoised`` '
            'loads ``rot_6_denoised.pt`` and ``acc_6_denoised.pt``; '
            '``denoised_fuse`` loads the files written by denoise.py.'
        ),
    )
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results_gid')
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location='cpu')


def build_test_all_features(
    loose_rot: torch.Tensor,
    loose_acc: torch.Tensor,
    tight_rot: torch.Tensor,
    tight_acc: torch.Tensor,
    imu_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same IMU preprocessing used by ``OurData.load_data``."""
    length = min(len(loose_rot), len(loose_acc), len(tight_rot), len(tight_acc))
    loose_rot = loose_rot[:length]
    tight_rot = tight_rot[:length]
    loose_acc = loose_acc[:length].reshape(length, imu_num, 3)
    tight_acc = tight_acc[:length].reshape(length, imu_num, 3)

    return OurData.build_imu_features(
        loose_rot, loose_acc, tight_rot, tight_acc,
    )


def per_sequence_mae(
    pred_features: torch.Tensor, tight_features: torch.Tensor, imu_num: int,
) -> tuple[torch.Tensor, int] | None:
    """Match test_all's window-end MAE calculation for one sequence."""
    # OurData creates ``T - SEQ_LEN`` windows. Their final indices are
    # therefore [SEQ_LEN - 1, T - 2], excluding the final source frame.
    pred_last = pred_features[SEQ_LEN - 1:-1].clone()
    tight_last = tight_features[SEQ_LEN - 1:-1]
    if len(pred_last) == 0:
        return None

    # Matches test_all.py: root acceleration is zeroed before MAE is measured.
    pred_last[:, -3:] = 0.0
    maes = torch.stack([
        (pred_last[:, imu * FEATURES_PER_IMU:(imu + 1) * FEATURES_PER_IMU]
         - tight_last[:, imu * FEATURES_PER_IMU:(imu + 1) * FEATURES_PER_IMU])
        .abs().mean()
        for imu in range(imu_num)
    ])
    return maes, len(pred_last)


def denoised_paths(sequence_dir: Path, imu_num: int, tag: str) -> tuple[Path, Path] | None:
    """Find a saved denoised pair with either numbered or generic filenames."""
    candidates = [(sequence_dir / f'rot_{imu_num}_{tag}.pt', sequence_dir / f'acc_{imu_num}_{tag}.pt')]
    candidates.append((sequence_dir / f'rot_{tag}.pt', sequence_dir / f'acc_{tag}.pt'))
    return next(((rot, acc) for rot, acc in candidates if rot.is_file() and acc.is_file()), None)


def evaluate_sequence(sequence_dir: Path, imu_num: int, denoise_tag: str) -> list[dict]:
    # Follow OurData.load_data exactly: use a numbered raw file only when it
    # actually exists (4-IMU recordings commonly use the unsuffixed names).
    suffix = f'_{imu_num}' if (sequence_dir / f'rot_{imu_num}.pt').is_file() else ''
    raw_paths = (sequence_dir / f'rot{suffix}.pt', sequence_dir / f'acc{suffix}.pt')
    tight_paths = (sequence_dir / f'vrot{suffix}.pt', sequence_dir / f'vacc{suffix}.pt')
    if not all(path.is_file() for path in (*raw_paths, *tight_paths)):
        return []

    raw_rot, raw_acc = map(load_tensor, raw_paths)
    tight_rot, tight_acc = map(load_tensor, tight_paths)
    tight_features = build_test_all_features(
        raw_rot, raw_acc, tight_rot, tight_acc, imu_num,
    )[1]

    variants = [('raw', raw_rot, raw_acc)]
    saved_pair = denoised_paths(sequence_dir, imu_num, denoise_tag)
    if saved_pair is not None:
        variants.append(('denoised', load_tensor(saved_pair[0]), load_tensor(saved_pair[1])))

    name, _, motion = sequence_dir.name.partition('_')
    rows = []
    for strategy, pred_rot, pred_acc in variants:
        pred_features, _ = build_test_all_features(
            pred_rot, pred_acc, tight_rot, tight_acc, imu_num,
        )
        result = per_sequence_mae(pred_features, tight_features, imu_num)
        if result is None:
            continue
        mae, samples = result
        row = {
            'name': name,
            'motion': motion or 'unknown',
            'strategy': strategy,
            'N_samples': samples,
        }
        row.update({f'IMU_{label}_MAE': float(value) for label, value in zip(IMU_CONFIGS[imu_num], mae)})
        row['IMU_mean_MAE'] = float(mae.mean())
        rows.append(row)
    return rows


def evaluate(data_root: Path, imu_num: int, denoise_tag: str) -> pd.DataFrame:
    if not data_root.is_dir():
        raise FileNotFoundError(f'Data root not found: {data_root}')
    rows = []
    for sequence_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        rows.extend(evaluate_sequence(sequence_dir, imu_num, denoise_tag))
    if not rows:
        raise RuntimeError('No complete raw IMU sequences were found.')
    return pd.DataFrame(rows).sort_values(['strategy', 'name', 'motion']).reset_index(drop=True)


def summarize(details: pd.DataFrame, imu_num: int) -> pd.DataFrame:
    metric_columns = [f'IMU_{name}_MAE' for name in IMU_CONFIGS[imu_num]] + ['IMU_mean_MAE']
    summary = []
    for strategy, group in details.groupby('strategy', sort=True):
        row = {
            'strategy': strategy,
            'sequences': len(group),
            'N_samples': int(group['N_samples'].sum()),
        }
        # This equal-weight sequence mean is exactly test_all.py's ALL row.
        row.update({column: float(group[column].mean()) for column in metric_columns})
        summary.append(row)
    return pd.DataFrame(summary)


def main():
    args = parse_args()
    details = evaluate(args.data_root, args.imu_num, args.denoise_tag)
    summary = summarize(details, args.imu_num)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.8f}'))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f'{args.data_root.name}_imu{args.imu_num}_{args.denoise_tag}_mae.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        details.to_excel(writer, sheet_name='PerSequence', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
