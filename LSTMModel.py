from Classifier import Classifier
from core import WordLSTMCore, CharLSTMCore
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.init as init
from tensorboard_logging import log_log_histogram


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

    def forward(self, inputs):
        '''
        char_embeddings = self.embedding_normalise(
            self.char_embedding_dropout(
                self.char_embedding(char_ids)
            )
        )
        word_embeddings = self.embedding_normalise(
            self.word_embedding_dropout(
                self.word_embedding(word_ids)
            )
        )
        '''
        # TODO id dropout sets some words to 'unknown'; further, the embedding for 'unknown' is used and trained
        # char_ids = self.char_id_dropout(inputs[0])
        # word_ids = self.word_id_dropout(inputs[1])

        char_ids = inputs[0]
        word_ids = inputs[1]

        first_ids = inputs[2]
        last_ids = inputs[3]

        char_embeddings = self.char_embedding(char_ids)
        word_embeddings = self.word_embedding(word_ids)

        char_dropout_embeddings = self.char_embedding_dropout(char_embeddings)
        word_dropout_embeddings = self.word_embedding_dropout(word_embeddings)

        # TODO hack, dropout skipped
        char_core_out = self.char_core(char_dropout_embeddings, first_ids, last_ids)
        word_core_out = self.word_core(word_dropout_embeddings)

        with torch.no_grad():
            catted = torch.cat(
                (
                    char_core_out,
                    word_core_out
                ),
                dim=1
            )
        meta_core_out = self.meta_core(catted)

        char_probs = self.char_classifier(char_core_out)
        word_probs = self.word_classifier(word_core_out)
        meta_probs = self.meta_classifier(meta_core_out)

        return char_probs, word_probs, meta_probs

    # sentence must separate words and punctuation by spaces
    # e.g.: 'Ich verstehe , dass man die Lücken normalerweise nicht lässt .'
    def tag_sentence(self, sentence):
        chars, words, firsts, lasts = [], [], [], []
        i = 0
        for word in sentence.split():
            chars.extend(word)
            chars.append(' ')
            words.append(word)
            firsts.append(i)
            i + len(word)
            lasts.append(i-1)
        chars = torch.tensor([chars[:-1]], dtype=torch.long, device=self.device)
        words = torch.tensor([words], dtype=torch.long, device=self.device)
        firsts = torch.tensor(firsts, dtype=torch.long, device=self.device)
        lasts = torch.tensor(lasts, dtype=torch.long, device=self.device)
        return [
            self.tags.get_value(index=tag_index)
            for tag_index
            in self.predict(chars, words, firsts, lasts)
        ]

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

    def log_char_net(self, writer, steps):
        log_log_histogram(
            writer=writer,
            steps=steps,
            name='grads/char_embedding',
            tensor=self.char_embedding.weight.grad
        )
        log_log_histogram(
            writer=writer,
            steps=steps,
            name='weights/char_embedding',
            tensor=self.char_embedding.weight,
        )
        self.char_core.log_tensorboard(
            writer=writer,
            name='char_core/',
            iteration_counter=steps
        )
        self.char_classifier.log_tensorboard(
            writer=writer,
            name='char_classifier/',
            iteration_counter=steps
        )

    def log_word_net(self, writer, steps):
        log_log_histogram(
            writer=writer,
            steps=steps,
            name='grads/word_embedding',
            tensor=self.word_embedding.weight.grad,
        )
        log_log_histogram(
            writer=writer,
            steps=steps,
            name='weights/word_embedding',
            tensor=self.word_embedding.weight,
        )
        self.word_core.log_tensorboard(
            writer=writer,
            name='word_core/',
            iteration_counter=steps
        )
        self.word_classifier.log_tensorboard(
            writer=writer,
            name='word_classifier/',
            iteration_counter=steps
        )

    def log_meta_net(self, writer, steps):
        self.meta_core.log_tensorboard(
            writer=writer,
            name='meta_core/',
            iteration_counter=steps
        )
        self.meta_classifier.log_tensorboard(
            writer=writer,
            name='meta_classifier/',
            iteration_counter=steps
        )

    def log_embeddings(self, writer, steps, word_list, char_list):
        print('save embeddings')
        writer.add_embedding(
            self.word_embedding.weight,
            global_step=steps,
            tag=f'word_embeddings{steps}',
            metadata=word_list
        )
        writer.add_embedding(
            self.char_embedding.weight,
            global_step=steps,
            tag=f'char_embeddings{steps}',
            metadata=char_list
        )
