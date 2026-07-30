r"""
PNP/GID upper-body pose evaluation.

This module retains the PNP evaluator's scalar rotation-angle convention while
reusing the project's primary :mod:`articulate` implementation.  It is kept
separate from ``evaluator.py`` because the latter's ``angle_between`` returns
axis-angle vectors for the other training/evaluation code.
"""

import torch

from .evaluator import BasePoseEvaluator
from .math.angular import (
    RotationRepresentation,
    angle_between,
    r6d_to_rotation_matrix,
    radian_to_degree,
)


__all__ = ['UpperMotionEvaluator', 'PoseEvaluator_PNP']


class UpperMotionEvaluator(BasePoseEvaluator):
    r"""PNP upper-body metric evaluator with the original metric layout."""

    def __init__(self, official_model_file: str, align_joint=0,
                 rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, position_mask=None,
                 angle_mask=None, device=torch.device('cpu')):
        super().__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = align_joint if isinstance(align_joint, int) else align_joint.value
        self.fps = fps
        self.position_mask = position_mask
        self.angle_mask = angle_mask
        self.sip_mask = torch.tensor([16, 17])

    @staticmethod
    def _scalar_rotation_angle(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
        """Match the legacy PNP angle helper's scalar output."""
        return angle_between(rot1, rot2).norm(dim=1)

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None,
                 tran_p=None, tran_t=None):
        f = self.fps
        pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(
            pose_local_p, shape_p, tran_p, calc_mesh=True)
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(
            pose_local_t, shape_t, tran_t, calc_mesh=True)

        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)
        lae = radian_to_degree(
            self._scalar_rotation_angle(pose_local_p, pose_local_t).view(pose_p.shape[0], -1))
        gae = radian_to_degree(
            self._scalar_rotation_angle(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)

        mje = je[:, self.position_mask] if self.position_mask is not None else torch.zeros(1)
        mlae = lae[:, self.angle_mask] if self.angle_mask is not None else torch.zeros(1)
        mgae = gae[:, self.angle_mask] if self.angle_mask is not None else torch.zeros(1)
        siplae = lae[:, self.sip_mask] if self.sip_mask is not None else torch.zeros(1)
        sipgae = gae[:, self.sip_mask] if self.sip_mask is not None else torch.zeros(1)

        return [
            [mlae.mean(dim=0), mlae.std(dim=0)],
            [mgae.mean(dim=0), mgae.std(dim=0)],
            [mje.mean(dim=0) * 100, mje.std(dim=0) * 100],
            [siplae.mean(dim=0), siplae.std(dim=0)],
            [sipgae.mean(dim=0), sipgae.std(dim=0)],
            [ve.mean(dim=0) * 100, ve.std(dim=0) * 100],
            [jkp.mean(dim=0) / 1000, jkp.std(dim=0) / 1000],
        ]


class PoseEvaluator_PNP:
    """Evaluate the ten upper-body joints output by the GID models."""

    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)',
             'Vertex Error (cm)', 'Jitter Error (km/s^3)']

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._base_motion_loss_fn = UpperMotionEvaluator(
            '../smpl/smpl/SMPL_NEUTRAL.pkl',
            device=device,
            position_mask=torch.tensor([3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21]),
            angle_mask=torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19]),
        )
        self.output_joint_mask = torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19])

    def __call__(self, pose_p, pose_t):
        p_full_body = torch.eye(3, device=pose_p.device).reshape(1, 1, 3, 3).repeat(
            len(pose_p), 24, 1, 1)
        p_full_body[:, self.output_joint_mask] = r6d_to_rotation_matrix(pose_p).reshape(
            -1, self.output_joint_mask.shape[0], 3, 3)
        t_full_body = torch.eye(3, device=pose_t.device).reshape(1, 1, 3, 3).repeat(
            len(pose_t), 24, 1, 1)
        t_full_body[:, self.output_joint_mask] = r6d_to_rotation_matrix(pose_t).reshape(
            -1, self.output_joint_mask.shape[0], 3, 3)
        return self._base_motion_loss_fn(pose_p=p_full_body, pose_t=t_full_body)
