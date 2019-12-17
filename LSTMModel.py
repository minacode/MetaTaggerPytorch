from core import WordLSTMCore, CharLSTMCore
from Classifier import Classifier
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_sequence
import torch.optim as optim


class BatchContainer:
    def __init__(self, chars, words, tags, firsts, lasts):
        self.chars = chars
        self.words = words
        self.tags = tags
        self.firsts = firsts
        self.lasts = lasts


class LSTMModel(nn.Module):
    def __init__(self, n_chars, n_words, n_tags, embedding_dim, debug=False, cuda=True):
        super(LSTMModel, self).__init__()

        self.debug = debug

        self.char_embedding = nn.Embedding(
            num_embeddings=n_chars,
            embedding_dim=embedding_dim)
        self.word_embedding = nn.Embedding(
            num_embeddings=n_words,
            embedding_dim=embedding_dim)

        self.char_embedding_dropout = nn.Dropout(p=0.05)
        self.word_embedding_dropout = nn.Dropout(p=0.05)

        self.char_core = CharLSTMCore(
            input_size=embedding_dim,
            n_lstm_layers=3,
            hidden_size=embedding_dim,
            dropout=0.33,
            debug=False)
        self.word_core = WordLSTMCore(
            input_size=embedding_dim,
            n_lstm_layers=3,
            hidden_size=embedding_dim,
            dropout=0.33)
        self.meta_core = WordLSTMCore(
            input_size=embedding_dim * 2,
            n_lstm_layers=1,
            hidden_size=embedding_dim * 2,
            dropout=0.33)

        self.char_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.word_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.meta_classifier = Classifier(
            input_size=embedding_dim * 2,
            n_tags=n_tags)

        self.char_loss = nn.CrossEntropyLoss(reduction='mean')
        self.word_loss = nn.CrossEntropyLoss(reduction='mean')
        self.meta_loss = nn.CrossEntropyLoss(reduction='mean')

        if cuda:
            self.device = torch.device('cuda')
            self.apply(lambda m: m.cuda())
        else:
            self.device = torch.device('cpu')

        self.char_optimizer = optim.Adam(
            list(self.char_embedding.parameters()) +
            list(self.char_core.parameters()) +
            list(self.char_classifier.parameters()))
        self.word_optimizer = optim.Adam(
            list(self.word_embedding.parameters()) +
            list(self.word_core.parameters()) +
            list(self.word_classifier.parameters()))
        self.meta_optimizer = optim.Adam(
            list(self.meta_core.parameters()) +
            list(self.meta_classifier.parameters()))

    def initialise(self):
        init.normal(self.char_embedding.parameters())
        init.zeros_(self.word_embedding.parameters())

        self.char_core.initialise()
        self.word_core.initialise()
        self.meta_core.initialise()

    def predict(self, chars, words, firsts, lasts):
        self.eval()

        char_embeddings = self.char_embedding(chars)
        word_embeddings = self.word_embedding(words)
        char_core_out = self.char_core(char_embeddings, firsts, lasts).unsqueeze(0)
        word_core_out = self.word_core(word_embeddings)
        catted = torch.cat((char_core_out, word_core_out), 2)
        meta_core_out = self.meta_core(catted)
        meta_probs = self.meta_classifier(meta_core_out).squeeze(dim=0)
        return meta_probs.argmax(dim=1)

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
        chars = torch.LongTensor([chars[:-1]])
        words = torch.LongTensor([words])
        firsts = torch.LongTensor(firsts)
        lasts = torch.LongTensor(lasts)
        return [
            self.tags.get_value(index=tag_index)
            for tag_index
            in self.predict(chars, words, firsts, lasts)
        ]

    def run_training(self, sentences, epochs=10, path=None):
        self.train()

        # load previous state if given
        if path is not None:
            dicts = torch.load(path)
            self.load_state_dicts(dicts)

        steps = 0
        for i in range(epochs):
            shuffle(sentences)
            for sentence in sentences:
                chars = torch.LongTensor(sentence['chars'], device=self.device)
                words = torch.LongTensor(sentence['words'], device=self.device)
                targets = torch.LongTensor(sentence['tags'], device=self.device)
                firsts = torch.LongTensor(sentence['firsts'], device=self.device)
                lasts = torch.LongTensor(sentence['lasts'], device=self.device)

                if self.debug:
                    print('chars', chars, 'words', words, 'targets', targets, 'firsts', firsts, 'lasts', lasts, sep='\n')

                self.char_optimizer.zero_grad()
                self.word_optimizer.zero_grad()
                self.meta_optimizer.zero_grad()

                char_embeddings = self.char_embedding_dropout(
                    self.char_embedding(chars))
                word_embeddings = self.word_embedding_dropout(
                    self.word_embedding(words))

                if self.debug:
                    print(char_embeddings.size(), word_embeddings.size())

                char_core_out = self.char_core(char_embeddings, firsts, lasts)
                word_core_out = self.word_core(word_embeddings)

                if self.debug:
                    print("pre cat", char_core_out.size(), word_core_out.size())

                catted = torch.cat((char_core_out, word_core_out), 2)

                if self.debug:
                    print("post cat", catted.size())

                meta_core_out = self.meta_core(catted)

                if self.debug:
                    print("meta out", meta_core_out.size())

                char_probs = self.char_classifier(char_core_out)  # .squeeze(dim=0)
                word_probs = self.word_classifier(word_core_out)  # .squeeze(dim=0)
                meta_probs = self.meta_classifier(meta_core_out)  # .squeeze(dim=0)

                if self.debug:
                    print("probs c w m t", char_probs.size(), word_probs.size(), meta_probs.size(), targets.size())

                targets = targets.permute(1, 0)
                char_probs = char_probs.permute(1, 2, 0)
                word_probs = word_probs.permute(1, 2, 0)
                meta_probs = meta_probs.permute(1, 2, 0)

                char_loss_out = self.char_loss(char_probs, targets)
                word_loss_out = self.word_loss(word_probs, targets)
                meta_loss_out = self.meta_loss(meta_probs, targets)

                if not steps % 100:
                    print(f'{steps}\t{char_loss_out.item()}\t{word_loss_out.item()}\t{meta_loss_out.item()}')

                char_loss_out.backward(retain_graph=True)
                self.char_optimizer.step()
                word_loss_out.backward(retain_graph=True)
                self.word_optimizer.step()
                meta_loss_out.backward()
                self.meta_optimizer.step()
                steps += 1

    def get_state_dicts(self):
        return {
            'model': self.state_dict(),
            'char_optimizer': self.char_optimizer.state_dict(),
            'word_optimizer': self.word_optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'dictionary': self.dictionary,
            'tags': self.tags,
            'language': self.language
        }

    # TODO write construction parameters, make static constructor from path
    def load_state_dicts(self, dicts, load_optimizers=True):
        self.load_state_dict(dicts['model'])
        self.dictionary = dicts['dictionary']
        self.tags = dicts['tags']
        self.language = dicts['language']
        if load_optimizers:
            self.char_optimizer.load_state_dict(dicts['char_optimizer'])
            self.word_optimizer.load_state_dict(dicts['word_optimizer'])
            self.meta_optimizer.load_state_dict(dicts['meta_optimizer'])

    def dev_eval(self):
        pass
        # TODO maybe use script supplied with data

    # TODO complete this
    def algorithm_1(self, train_data):
        self.initialze()
        epochs = 10

        best_f1 = 0

        for _ in range(epochs):
            char_logits, char_preds = self.run_char_model(train_data)
            self.char_optimizer.step()
            word_logits, word_preds = self.run_word_model(train_data)
            self.word_optimizer.step()
            meta_logits, meta_preds = self.run_meta_model(train_data)
            self.meta_optimizer.step()

            f1 = self.dev_eval()
            if f1 > best_f1:
                # implement this
                self.lock_best_model()
                best_f1 = f1
