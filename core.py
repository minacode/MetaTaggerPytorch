import torch
import torch.nn as nn
import torch.nn.init as init
from tensorboard_logging import log_log_histogram


class WordLSTMCore(nn.Module):
    def __init__(self, input_size, n_lstm_layers, hidden_size, dropout, residual):
        super(WordLSTMCore, self).__init__()

        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.residual = residual

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True)
        # TODO hack, before: check bias=True (by default)
        self.linear = nn.Linear(
            in_features=hidden_size*2,
            out_features=self.output_size(),
            bias=True
        )

    # TODO what size is this really?
    def output_size(self):
        return self.hidden_size

    def initialise(self):
        # TODO this is not documented in the paper
        # TODO initialise bilstm weights
        init.normal_(self.linear.weight)

    def forward(self, x):
        unsqueezed_x = torch.unsqueeze(x, dim=1)
        lstm_out, _ = self.bilstm(
            unsqueezed_x
        )
        squeezed_lstm_out = lstm_out.squeeze(dim=1)

        # TODO residual instead of paper
        if self.residual:
            linear_in = squeezed_lstm_out + torch.cat([x, x], dim=1)
        else:
            linear_in = squeezed_lstm_out

        linear_out = self.linear(linear_in)
        return linear_out

    def log_tensorboard(self, writer, name, iteration_counter):
        for a in ['bias', 'weight']:
            for b in ['ih', 'hh']:
                for direction in ['', '_reverse']:
                    for i in range(self.n_lstm_layers):
                        log_log_histogram(
                            writer=writer,
                            steps=iteration_counter,
                            name=f'grads/{name}bilstm_combined',
                            tensor=getattr(self.bilstm, f'{a}_{b}_l{i}{direction}').grad,
                        )
                        log_log_histogram(
                            writer=writer,
                            steps=iteration_counter,
                            name=f'weights/{name}bilstm_combined',
                            tensor=getattr(self.bilstm, f'{a}_{b}_l{i}{direction}'),
                        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'grads/{name}linear',
            tensor=self.linear.weight.grad
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'grads/{name}linear/bias',
            tensor=self.linear.bias.grad
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'weights/{name}linear',
            tensor=self.linear.weight
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'weights/{name}linear/bias',
            tensor=self.linear.bias
        )


class CharLSTMCore(nn.Module):
    def __init__(self, input_size, n_lstm_layers, hidden_size, dropout, residual, debug=False):
        super(CharLSTMCore, self).__init__()

        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.residual = residual

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True)
        self.linear = nn.Linear(
            in_features=hidden_size*4,
            out_features=self.output_size())
        self.debug = debug

    # TODO what size is this really?
    def output_size(self):
        return self.hidden_size

    def initialise(self):
        # TODO init bilstm_weights
        init.normal_(self.linear.weight)

    def forward(self, x, firsts, lasts):
        unsqueezed_x = torch.unsqueeze(x, dim=1)
        lstm_out, _ = self.bilstm(
            unsqueezed_x
        )
        squeezed_lstm_out = lstm_out.squeeze(dim=1)

        # TODO this is residual, other than paper
        if self.residual:
            residual_out = squeezed_lstm_out + torch.cat([x, x], dim=1)
        else:
            residual_out = squeezed_lstm_out

        catted = torch.cat(
            [residual_out[firsts], residual_out[lasts]],
            dim=1
        )

        linear_out = self.linear(catted)
        return linear_out

    def log_tensorboard(self, writer, name, iteration_counter):
        for a in ['bias', 'weight']:
            for b in ['ih', 'hh']:
                for direction in ['', '_reverse']:
                    for i in range(self.n_lstm_layers):
                        log_log_histogram(
                            writer=writer,
                            steps=iteration_counter,
                            name=f'grads/{name}bilstm_combined',
                            tensor=getattr(self.bilstm, f'{a}_{b}_l{i}{direction}').grad,
                        )
                        log_log_histogram(
                            writer=writer,
                            steps=iteration_counter,
                            name=f'weights/{name}bilstm_combined',
                            tensor=getattr(self.bilstm, f'{a}_{b}_l{i}{direction}'),
                        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'grads/{name}linear',
            tensor=self.linear.weight.grad
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'grads/{name}linear/bias',
            tensor=self.linear.bias.grad
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'weights/{name}linear',
            tensor=self.linear.weight
        )
        log_log_histogram(
            writer=writer,
            steps=iteration_counter,
            name=f'weights/{name}linear/bias',
            tensor=self.linear.bias
        )
