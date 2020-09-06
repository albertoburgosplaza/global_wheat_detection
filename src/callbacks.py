import numpy as np
import torch


# from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if metric doesn't improve after a given patience."""
    def __init__(self, patience=7, mode="min", verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time metric improved.
                            Default: 7
            mode (str): Sets id the improvement is due to a maximization ("max") or a minimization ("min").
                            Default: "min"
            verbose (bool): If True, prints a message for each metric improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, metric, model):

        if self.mode == "min":
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when metric improves.'''
        if self.verbose:
            print(f'Validation metric improved ({self.metric_min:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.metric_min = metric