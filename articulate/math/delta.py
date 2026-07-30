import torch
import numpy as np

# ---------- 辅助：把 (N,48) -> R (N,4,3,3) and A (N,4,3) ----------
def _split_imu_tensor(imu, first='rot'):
    """
    imu: (N,48)
    first: 'rot' or 'acc'  — 表示每个 imu 内部是 [rot(9), acc(3)] 还是 [acc(3), rot(9)]
    returns: R (N,4,3,3), A (N,4,3)
    """
    assert imu.dim() == 2 and imu.size(1) == 48
    N = imu.size(0)
    imu4 = imu.view(N, 4, 12)  # (N,4,12): per imu 12 dims
    if first == 'rot':
        rot_flat = imu4[:, :, :9]   # (N,4,9)
        acc = imu4[:, :, 9:]        # (N,4,3)
    else:  # 'acc'
        acc = imu4[:, :, :3]
        rot_flat = imu4[:, :, 3:]
    R = rot_flat.view(N, 4, 3, 3)
    return R, acc

def _pack_imu_tensor(R, A, first='rot'):
    """
    R: (N,4,3,3), A: (N,4,3) -> return (N,48)
    """
    N = R.shape[0]
    rot_flat = R.reshape(N, 4, 9)
    if first == 'rot':
        imu4 = torch.cat([rot_flat, A], dim=-1)  # (N,4,12)
    else:
        imu4 = torch.cat([A, rot_flat], dim=-1)
    return imu4.reshape(N, 48)

# ---------- SVD 投影到 SO(3)（批量、无 in-place、可导） ----------
def _project_to_SO3_torch(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3)
    返回: 同形状，逐样本投影到最近的 SO(3)
    """
    orig_shape = R.shape
    M = R.reshape(-1, 3, 3)  # [B,3,3]

    # full_matrices=False 更高效；SVD 对奇异值退化点处梯度可能不稳定，这是 SVD 本身的局限
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U @ diag(S) @ Vh

    # 先得到最近的 O(3)（可能 det=-1）
    R_o = U @ Vh  # [B,3,3]

    # 计算每个样本的符号修正 s = sign(det(UV^T))，并构造批量对角修正矩阵 S_fix
    det_o = torch.det(R_o)  # [B]
    s = torch.where(det_o < 0, R_o.new_tensor(-1.0), R_o.new_tensor(1.0))  # [B]

    # S_fix = diag(1,1,s) (批量)
    B = M.shape[0]
    S_fix = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,3,3]
    S_fix[:, 2, 2] = s  # 非 in-place 对计算图无害（S_fix 是新张量）

    # 应用修正，得到 SO(3)
    R_so = U @ S_fix @ Vh  # [B,3,3]

    return R_so.reshape(orig_shape)

# ---------- 主函数：计算 delta ----------
def compute_delta(imu_loose, imu_tight, first='rot', project=True):
    """
    Compute a physically-meaningful delta such that:
      rot part: R_delta = R_tight @ R_loose^T  (relative rotation)
      acc part: a_diff = a_loose - a_tight
    Inputs:
      imu_loose, imu_tight: torch.Tensor shape (N,48)
      first: 'rot' or 'acc' (which field appears first inside each 12-d IMU block)
      project: whether to SVD-project R_rel to valid rotation matrices
    Returns:
      delta: (N,48) in the same IMU packing: per IMU [rot_rel_flatten(9), acc_diff(3)] if first=='rot'
             or [acc_diff(3), rot_rel_flatten(9)] if first=='acc'
    """
    assert imu_loose.shape == imu_tight.shape
    Rl, Al = _split_imu_tensor(imu_loose, first=first)
    Rt, At = _split_imu_tensor(imu_tight, first=first)

    # Compute relative rotations: R_rel = R_t @ R_l^T
    Rl_T = Rl.transpose(-1, -2)  # (N,4,3,3)
    R_rel = torch.matmul(Rt, Rl_T)  # (N,4,3,3)

    if project:
        R_rel = _project_to_SO3_torch(R_rel)

    # acceleration difference: a_loose - a_tight
    a_diff = Al - At  # (N,4,3)

    # pack back: flatten rot and acc into (N,48)
    delta = _pack_imu_tensor(R_rel, a_diff, first=first)
    return delta

# ---------- 主函数：从 delta 重构 imu_tight ----------
def reconstruct_tight_from_delta(imu_loose, delta, first='rot', project=True):
    """
    Given imu_loose and delta (as computed by compute_delta),
    reconstruct imu_tight (N,48).
    rot reconstruction: R_tight = R_delta @ R_loose
    acc reconstruction: a_tight = a_loose - a_diff
    """

    shape_0, shape_1 = imu_loose.shape[0], imu_loose.shape[1]
    Rl, Al = _split_imu_tensor(imu_loose.reshape(-1, 48), first=first)
    R_rel, a_diff = _split_imu_tensor(delta.reshape(-1, 48), first=first)

    # Optionally ensure R_rel is a proper rotation
    if project:
        R_rel = _project_to_SO3_torch(R_rel)

    # reconstruct
    R_tight = torch.matmul(R_rel, Rl)
    A_tight = Al - a_diff

    imu_tight = _pack_imu_tensor(R_tight, A_tight, first=first)
    return imu_tight.reshape(shape_0, -1, 48)
