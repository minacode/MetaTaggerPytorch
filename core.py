import torch
import torch.nn as nn
import torch.nn.init as init


class WordLSTMCore(nn.Module):
    def __init__(self, input_size, n_lstm_layers, hidden_size, dropout):
        super(WordLSTMCore, self).__init__()

        self.hidden_size = hidden_size

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True)
        self.linear = nn.Linear(
            in_features=hidden_size*2,
            out_features=self.output_size())

    # TODO what size is this really?
    def output_size(self):
        return self.hidden_size

    def initialise(self):
        # TODO this is not documented in the paper
        init.normal(self.linear.parameters())

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        x = self.linear(lstm_out)
        return x


class CharLSTMCore(nn.Module):
    def __init__(self, input_size, n_lstm_layers, hidden_size, dropout, debug=False):
        super(CharLSTMCore, self).__init__()

        self.hidden_size = hidden_size

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
        init.normal(self.linear.parameters())

    def forward(self, x, firsts, lasts):
        if self.debug:
            print(
                f'input {x.size()}\n'
                # f'{x}\n'
                f'firsts {firsts.size()}\n'
                # f'{firsts}\n'
                f'lasts {lasts.size()}\n'
                # f'{lasts}'
            )
        lstm_out, _ = self.bilstm(x)
        if self.debug:
            print(
                f'lstm_out {lstm_out.size()}\n'
                # f'{lstm_out}'
            )
        catted = torch.cat(
            [lstm_out[0, firsts], lstm_out[0, lasts]],
            dim=2
        )
        if self.debug:
            print(
                f'catted {catted.size()}\n'
                # f'{catted}'
            )
        # TODO this is definitely the wrong shape
        x = self.linear(catted)
        if self.debug:
            print(
                f'output {x.size()}\n'
                # f'{x}'
            )
        return x
