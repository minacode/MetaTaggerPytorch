import torch.nn as nn
from tensorboard_logging import log_log_histogram


class Classifier(nn.Module):
    def __init__(self, input_size, n_tags, debug=False):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(
            in_features=input_size,
            out_features=n_tags
        )
        self.elu = nn.ELU()
        self.debug = debug

    def forward(self, x):
        if self.debug:
            print(f'pre linear elu {x.size()}')
        x = self.linear(x)
        if self.debug:
            print(f'pre elu {x.size()}')
        x = self.elu(x)
        if self.debug:
            print(f'post elu {x.size()}')
        return x

    def log_tensorboard(self, writer, name, iteration_counter):
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'grads/{name}linear',
            tensor=self.linear.weight.grad
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'weights/{name}linear',
            tensor=self.linear.weight
        )

