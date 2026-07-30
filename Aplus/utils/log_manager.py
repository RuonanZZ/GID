import os
import pandas as pd
import torch


class LogManager:
    def __init__(self, items: list, log_data=None):
        """

        :param items: e.g. ['epoch', 'loss_train', 'loss_eval']
        :param log_data: Dict
        """
        if log_data is not None:
            self.log = log_data
        else:
            self.log = {}
            for item in items:
                self.log.update({item:[]})

    def update(self, values:dict):
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.cpu()
                if value.numel() == 1:
                    value = float(value)
            self.log[key].append(value)

    def print_latest(self):
        for key, value in self.log.items():
            last_value = self.log[key][-1]
            if type(last_value) == int or type(last_value) == float:
                last_value = round(last_value, 5)
            print(f'| {key}: {last_value } ', end='')
        print('|')

    def align_lengths(self):
        L = max((len(v) for v in self.log.values()), default=0)
        for k, v in self.log.items():
            if len(v) < L:
                v.extend([None] * (L - len(v)))

    def to_excel(self, path):
        import pandas as pd
        self.align_lengths()                      # <--- 关键：先对齐
        df_log = pd.DataFrame.from_dict(self.log)
        path_parent = os.path.dirname(path)
        if path_parent and not os.path.exists(path_parent):
            os.makedirs(path_parent, exist_ok=True)
        df_log.to_excel(path, index=False)

    def load_data(self, data:dict):
        self.log = data

    @classmethod
    def from_dict(cls, data:dict):
        return cls(items=[], log_data=data)

    @classmethod
    def from_excel(cls, path):
        df = pd.read_excel(path)
        return cls(items=[], log_data=df.to_dict())

