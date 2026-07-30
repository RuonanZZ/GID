import torch

from .base_models import BaseModel


class BiPoser(BaseModel):
    def __init__(self, net_s1, net_s2, export_mode=False):
        super(BiPoser, self).__init__()

        self.net_s1 = net_s1
        self.net_s2 = net_s2
        self.export_mode = export_mode

    def forward_train(self, x, *args):

        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2 = args
            out_1, h_s1, c_s1 = self.net_s1(x, h_s1, c_s1)
            out_2, h_s2, c_s2 = self.net_s2(torch.cat([x, out_1], dim=-1), h_s2, c_s2)
        else:
            out_1 = self.net_s1(x)
            out_2 = self.net_s2(torch.cat([x, out_1], dim=-1))

        if len(args) > 0:
            return out_1, out_2, h_s1, c_s1, h_s2, c_s2
        return out_1, out_2

    def joint_out(self, x):
        out_1 = self.net_s1(x)
        return out_1

    def pose_out(self, x, *args):
        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2 = args
            out_1, h_s1, c_s1 = self.net_s1(x, h_s1, c_s1)
            out_2, h_s2, c_s2 = self.net_s2(torch.cat([x, out_1], dim=-1), h_s2, c_s2)
        else:
            out_1 = self.net_s1(x)
            out_2 = self.net_s2(torch.cat([x, out_1], dim=-1))

        if len(args) > 0:
            return out_2, h_s1, c_s1, h_s2, c_s2
        return out_2

    def forward(self, acc_cat_rot, *args):
        x = acc_cat_rot
        if self.export_mode:
            return self.pose_out(x, *args)
        return self.forward_train(x, *args)