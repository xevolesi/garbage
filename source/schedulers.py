import math
from torch.optim.lr_scheduler import MultiStepLR


class WarmupMultiStepLR:
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_iters=1500,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        last_epoch=-1
    ) -> None:
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        
        # Сохраняем базовые learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Создаем стандартный MultiStepLR для основного расписания
        self.scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
        
        self.last_epoch = last_epoch
        self.current_iter = 0
        
    def get_warmup_lr(self, cur_iter):
        if cur_iter < self.warmup_iters:
            alpha = cur_iter / self.warmup_iters
            warmup_factor = self.warmup_ratio * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return None
    
    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        if self.warmup_by_epoch:
            warmup_lrs = self.get_warmup_lr(self.last_epoch)
        else:
            warmup_lrs = self.get_warmup_lr(self.current_iter)

        if warmup_lrs is not None:
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group['lr'] = lr
        else:
            self.scheduler.step(self.last_epoch)

    def step_iter(self):
        self.current_iter += 1
        if not self.warmup_by_epoch and self.current_iter <= self.warmup_iters:
            warmup_lrs = self.get_warmup_lr(self.current_iter)
            if warmup_lrs is not None:
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                    param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'current_iter': self.current_iter,
            'scheduler_state': self.scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.current_iter = state_dict['current_iter']
        self.scheduler.load_state_dict(state_dict['scheduler_state'])