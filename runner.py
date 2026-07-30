"""Minimal training runner used by the GID training scripts."""

from tqdm import tqdm
from torch.utils.data import DataLoader

from Aplus.runner import BaseTrainer
from Aplus.utils import DataMeter


class MyTrainer(BaseTrainer):
    """GID trainer with one optional validation evaluator."""

    def run(self, epoch, data_shuffle=True, evaluator=None, verbose=False, batch_sampler=None):
        if batch_sampler is None:
            data_loader = DataLoader(
                dataset=self.data,
                batch_size=self.batch_size,
                shuffle=data_shuffle,
                drop_last=False,
            )
        else:
            data_loader = DataLoader(dataset=self.data, batch_sampler=batch_sampler, drop_last=False)

        device = self.get_model_device()
        loss_meter = DataMeter()
        for _ in range(epoch):
            loss_meter.reset()
            self.model.train()
            for step, (inputs, targets) in enumerate(tqdm(data_loader, leave=False)):
                self.optimizer.zero_grad()
                prediction = self.model(inputs.to(device))
                loss = self.loss_func(prediction, targets.to(device))
                loss.backward()
                self.optimizer.step()
                loss_meter.update(value=loss.item(), n_sample=len(targets))
                if verbose:
                    print(f'\riter {step} | {len(self.data) // self.batch_size}', end='')
            self.epoch += 1

        loss_eval = evaluator.run() if evaluator is not None else -1
        self.log_manager.update({
            'epoch': self.epoch,
            'loss_train': loss_meter.get_avg(),
            'loss_eval': loss_eval,
        })
        self.log_manager.print_latest()
