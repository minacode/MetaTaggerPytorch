from Classifier import Classifier
from core import WordLSTMCore, CharLSTMCore
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.init as init


class LSTMModel(nn.Module):
    def __init__(self, n_chars, n_words, n_tags, embedding_dim, residual, cuda, debug=False):
        super(LSTMModel, self).__init__()

        self.debug = debug

        self.residual = residual

        self.char_id_dropout = nn.Dropout(p=0.05)
        self.word_id_dropout = nn.Dropout(p=0.05)

        self.char_embedding = nn.Embedding(
            num_embeddings=n_chars,
            embedding_dim=embedding_dim)
        self.word_embedding = nn.Embedding(
            num_embeddings=n_words,
            embedding_dim=embedding_dim)

        self.embedding_normalise = nn.Softmax(dim=2)

        self.char_embedding_dropout = nn.Dropout(p=0.05)
        self.word_embedding_dropout = nn.Dropout(p=0.05)

        self.char_core = CharLSTMCore(
            input_size=embedding_dim,
            n_lstm_layers=3,
            hidden_size=embedding_dim,
            dropout=0.05,  # 0.33,
            debug=debug,
            residual=residual
        )
        self.word_core = WordLSTMCore(
            input_size=embedding_dim,
            n_lstm_layers=3,
            hidden_size=embedding_dim,
            dropout=0.05,  # 0.33,
            residual=residual
        )
        self.meta_core = WordLSTMCore(
            input_size=embedding_dim * 2,
            n_lstm_layers=1,
            hidden_size=embedding_dim * 2,
            dropout=0.05,  # 0.33,
            residual=residual
        )

        self.char_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.word_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.meta_classifier = Classifier(
            input_size=embedding_dim * 2,
            n_tags=n_tags)

        if cuda:
            self.device = torch.device('cuda')
            self.apply(lambda m: m.cuda())
        else:
            self.device = torch.device('cpu')

    def initialise(self):
        print('initialise LSTMModel')
        init.uniform_(self.char_embedding.weight)
        init.uniform_(self.word_embedding.weight)

        self.char_core.initialise()
        self.word_core.initialise()
        self.meta_core.initialise()

    def forward_char_net(self, inputs):
        ids, first_ids, last_ids = inputs
        embeddings = self.char_embedding(ids)
        dropout_embeddings = self.char_embedding_dropout(embeddings)
        core_out = self.char_core(dropout_embeddings, first_ids, last_ids)
        return core_out

    def forward_word_net(self, ids):
        embeddings = self.word_embedding(ids)
        dropout_embeddings = self.word_embedding_dropout(embeddings)
        core_out = self.word_core(dropout_embeddings)
        return core_out

    # inputs of the form (char_net_out, word_net_out)
    def forward_meta_net(self, inputs):
        char_net_out, word_net_out = inputs
        with torch.no_grad():
            catted = torch.cat(
                (
                    char_net_out,
                    word_net_out
                ),
                dim=1
            )
        core_out = self.meta_core(catted)
        return core_out

    # inputs is of the form (char_ids, first_ids, last_ids)
    def get_char_probabilities(self, inputs):
        net_out = self.forward_char_net(inputs)
        probs = self.char_classifier(net_out)
        return probs

    def get_word_probabilities(self, ids):
        net_out = self.forward_word_net(ids)
        probs = self.word_classifier(net_out)
        return probs

    # inputs of the form (char_ids, word_ids, first_ids, last_ids)
    def get_meta_probabilities(self, inputs):
        char_ids, word_ids, first_ids, last_ids = inputs

        char_net_out = self.forward_char_net((char_ids, first_ids, last_ids))
        word_net_out = self.forward_word_net(word_ids)
        meta_net_out = self.forward_meta_net((char_net_out, word_net_out))
        meta_probs = self.meta_classifier(meta_net_out)
        return meta_probs

    # inputs of the form (char_ids, word_ids, first_ids, last_ids)
    def forward(self, inputs):
        return self.get_meta_probabilities(inputs)

    def get_char_params(self):
        return chain(
            self.char_embedding.parameters(),
            self.char_core.parameters(),
            self.char_classifier.parameters()
        )

    def get_word_params(self):
        return chain(
            self.word_embedding.parameters(),
            self.word_core.parameters(),
            self.word_classifier.parameters()
        )

    def get_meta_params(self):
        return chain(
            self.meta_core.parameters(),
            self.meta_classifier.parameters()
        )

    def get_state_dicts(self, language, dataset):
        return {
            'model': self.state_dict(),
            'char_optimizer': self.char_optimizer.state_dict(),
            'word_optimizer': self.word_optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'language': language,
            'dataset': dataset
        }

    # TODO write construction parameters, make static constructor from path
    def load_state_dicts(self, dicts, load_optimizers=True):
        self.load_state_dict(dicts['model'])
        if load_optimizers:
            self.char_optimizer.load_state_dict(dicts['char_optimizer'])
            self.word_optimizer.load_state_dict(dicts['word_optimizer'])
            self.meta_optimizer.load_state_dict(dicts['meta_optimizer'])
