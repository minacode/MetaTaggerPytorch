from core import WordLSTMCore, CharLSTMCore
from Classifier import Classifier
from Corpora.ud_test_v2_0_conll2017.evaluation_script.conll17_ud_eval import evaluate, load_conllu_file
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_sequence
import torch.optim as optim
from build_dicts import ID, FORM, tag_name_to_column
from tensorboardX import SummaryWriter


class BatchContainer:
    def __init__(self, chars, words, tags, firsts, lasts):
        self.chars = chars
        self.words = words
        self.tags = tags
        self.firsts = firsts
        self.lasts = lasts


class LSTMModel(nn.Module):
    def __init__(self, n_chars, n_words, n_tags, embedding_dim, cuda, debug=False):
        super(LSTMModel, self).__init__()

        self.debug = debug

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
            n_lstm_layers=2,
            hidden_size=embedding_dim,
            dropout=0.1,  # 0.33,
            debug=debug)
        self.word_core = WordLSTMCore(
            input_size=embedding_dim,
            n_lstm_layers=1,
            hidden_size=embedding_dim,
            dropout=0.1)  # 0.33)
        self.meta_core = WordLSTMCore(
            input_size=embedding_dim * 2,
            n_lstm_layers=1,
            hidden_size=embedding_dim * 2,
            dropout=0.1)  # 0.33)

        self.char_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.word_classifier = Classifier(
            input_size=embedding_dim,
            n_tags=n_tags)
        self.meta_classifier = Classifier(
            input_size=embedding_dim * 2,
            n_tags=n_tags)

        self.char_loss = nn.MSELoss(reduction='sum')
        self.word_loss = nn.MSELoss(reduction='sum')
        self.meta_loss = nn.MSELoss(reduction='sum')

        if cuda:
            self.device = torch.device('cuda')
            self.apply(lambda m: m.cuda())
        else:
            self.device = torch.device('cpu')

        optimizer_args = {
            'lr': 0.2,  # TODO test other values
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.999994
        }

        self.char_optimizer = optim.Adam(
            list(self.char_embedding.parameters()) +
            list(self.char_core.parameters()) +
            list(self.char_classifier.parameters()),
            **optimizer_args
        )
        self.word_optimizer = optim.Adam(
            list(self.word_embedding.parameters()) +
            list(self.word_core.parameters()) +
            list(self.word_classifier.parameters()),
            **optimizer_args
        )
        self.meta_optimizer = optim.Adam(
            list(self.meta_core.parameters()) +
            list(self.meta_classifier.parameters()),
            **optimizer_args
        )

        gamma = 0.999
        self.char_optimizer_decay = optim.lr_scheduler.ExponentialLR(
            optimizer=self.char_optimizer,
            gamma=gamma
        )
        self.word_optimizer_decay = optim.lr_scheduler.ExponentialLR(
            optimizer=self.word_optimizer,
            gamma=gamma
        )
        self.meta_optimizer_decay = optim.lr_scheduler.ExponentialLR(
            optimizer=self.meta_optimizer,
            gamma=gamma
        )

    def initialise(self):
        print('initialise LSTMModel')
        init.uniform_(self.char_embedding.weight)
        init.uniform_(self.word_embedding.weight)

        self.char_core.initialise()
        self.word_core.initialise()
        self.meta_core.initialise()

    def forward(self, char_ids, word_ids, first_ids, last_ids):
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
        char_embeddings = self.char_embedding_dropout(
            self.char_embedding(char_ids)
        )
        word_embeddings = self.word_embedding_dropout(
            self.word_embedding(word_ids)
        )

        char_core_out = self.char_core(char_embeddings, first_ids, last_ids)
        word_core_out = self.word_core(word_embeddings)
        catted = torch.cat(
            (
                char_core_out,
                word_core_out
            ),
            dim=1
        )
        meta_core_out = self.meta_core(catted)
        char_probs = self.char_classifier(char_core_out)  # .squeeze(dim=0)
        word_probs = self.word_classifier(word_core_out)  # .squeeze(dim=0)
        meta_probs = self.meta_classifier(meta_core_out)  # .squeeze(dim=0)
        # char_probs = char_probs.permute(1, 2, 0)
        # word_probs = word_probs.permute(1, 2, 0)
        # meta_probs = meta_probs.permute(1, 2, 0)

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
        chars = torch.LongTensor([chars[:-1]])
        words = torch.LongTensor([words])
        firsts = torch.LongTensor(firsts)
        lasts = torch.LongTensor(lasts)
        return [
            self.tags.get_value(index=tag_index)
            for tag_index
            in self.predict(chars, words, firsts, lasts)
        ]

    def run_training(self, sentences, tag_name, n_tags, word_list, epochs=10, path=None):
        self.train()

        writer = SummaryWriter()

        n_sentences = len(sentences)

        # load previous state if given
        if path is not None:
            dicts = torch.load(path)
            self.load_state_dicts(dicts)

        steps = 0
        for n_epoch in range(epochs):
            print(f'starting epoch {n_epoch}')
            shuffle(sentences)
            for sentence in sentences:
                self.char_optimizer.zero_grad()
                self.word_optimizer.zero_grad()
                self.meta_optimizer.zero_grad()

                chars = torch.LongTensor(sentence['char_ids']).to(self.device)
                words = torch.LongTensor(sentence['word_ids']).to(self.device)
                # targets = torch.LongTensor(sentence['tag_ids'][tag_name]).to(self.device)
                targets = [
                    [0 for _ in range(n_tags)]
                    for _ in range(len(sentence['tag_ids'][tag_name]))
                ]
                for n_tag, tag_id in enumerate(sentence['tag_ids'][tag_name]):
                    targets[n_tag][tag_id] = 1
                targets = torch.Tensor(targets).to(self.device)
                firsts = torch.LongTensor(sentence['first_ids']).to(self.device)
                lasts = torch.LongTensor(sentence['last_ids']).to(self.device)

                char_probs, word_probs, meta_probs = self.forward(
                    char_ids=chars,
                    word_ids=words,
                    first_ids=firsts,
                    last_ids=lasts
                )

                if not steps % 1000:
                    print('save embeddings')
                    writer.add_embedding(
                        self.word_embedding.weight,
                        global_step=steps,
                        tag=f'word_embeddings{steps}',
                        metadata=word_list
                    )

                # targets = targets.permute(1, 0)

                char_one_loss = (1 - char_probs.sum(dim=1)).sum(dim=0)
                word_one_loss = (1 - word_probs.sum(dim=1)).sum(dim=0)
                meta_one_loss = (1 - meta_probs.sum(dim=1)).sum(dim=0)

                # char_loss_out = ((char_probs - targets).sum() + char_one_loss).abs()
                # word_loss_out = ((word_probs - targets).sum() + word_one_loss).abs()
                # meta_loss_out = ((meta_probs - targets).sum() + meta_one_loss).abs()

                char_loss_out = self.char_loss(char_probs, targets)
                word_loss_out = self.word_loss(word_probs, targets)
                meta_loss_out = self.meta_loss(meta_probs, targets)

                char_loss_out += char_one_loss
                word_loss_out += word_one_loss
                meta_loss_out += meta_one_loss

                '''
                char_loss_out = 0
                word_loss_out = 0
                meta_loss_out = 0
                for i, target in enumerate(targets):
                    for j in range(char_probs.shape[1]):
                        if target == j:
                            char_loss_out += (1 - char_probs[i, target])
                            word_loss_out += (1 - word_probs[i, target])
                            meta_loss_out += (1 - meta_probs[i, target])
                        else:
                            char_loss_out += (char_probs[i, target])
                            word_loss_out += (word_probs[i, target])
                            meta_loss_out += (meta_probs[i, target])
                '''

                # log losses
                writer.add_scalar('losses/char', char_loss_out, steps)
                writer.add_scalar('losses/word', word_loss_out, steps)
                writer.add_scalar('losses/meta', meta_loss_out, steps)

                if not steps % 100:
                    print(f'{steps}/{n_sentences}\t{char_loss_out.item()}\t{word_loss_out.item()}\t{meta_loss_out.item()}')

                char_loss_out.backward(retain_graph=True)
                word_loss_out.backward(retain_graph=True)
                meta_loss_out.backward()

                if self.debug and not steps % 1000:
                    print(word_probs)
                    print(word_probs.argmax(dim=1))
                    print(targets)
                    print(char_one_loss.item(), word_one_loss.item(), meta_one_loss.item())
                    print(self.word_embedding.weight.grad)

                if not steps % 10:
                    # log probabilities
                    writer.add_histogram('probabilities/chars', char_probs, steps)
                    writer.add_histogram('probabilities/words', word_probs, steps)
                    writer.add_histogram('probabilities/meta', meta_probs, steps)

                    writer.add_histogram(
                        'char_embedding_grads',
                        self.char_embedding.weight.grad,
                        steps
                    )
                    writer.add_histogram(
                        'word_embedding_grads',
                        self.word_embedding.weight.grad,
                        steps
                    )

                    self.char_core.log_tensorboard(
                        writer=writer,
                        name='char_core/',
                        iteration_counter=steps
                    )
                    self.word_core.log_tensorboard(
                        writer=writer,
                        name='word_core/',
                        iteration_counter=steps
                    )
                    self.meta_core.log_tensorboard(
                        writer=writer,
                        name='meta_core/',
                        iteration_counter=steps
                    )

                    self.char_classifier.log_tensorboard(
                        writer=writer,
                        name='char_classifier/',
                        iteration_counter=steps)
                    self.word_classifier.log_tensorboard(
                        writer=writer,
                        name='word_classifier/',
                        iteration_counter=steps)
                    self.meta_classifier.log_tensorboard(
                        writer=writer,
                        name='meta_classifier/',
                        iteration_counter=steps)

                self.char_optimizer.step()
                self.word_optimizer.step()
                self.meta_optimizer.step()

                writer.add_scalar(
                    'lr',
                    self.char_optimizer_decay.get_lr(),
                    steps
                )

                # self.char_optimizer_decay.step(epoch=steps)
                # self.word_optimizer_decay.step(epoch=steps)
                # self.meta_optimizer_decay.step(epoch=steps)

                steps += 1
        writer.close()

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

    def dev_eval(self, tag_name, path, labeled_data):
        work_data = load_conllu_file(path)
        words_count = len(work_data.tokens)

        tag_column_id = tag_name_to_column(tag_name=tag_name)

        print(f'words: {words_count}')

        # TODO make a function ouf of this that is also called in build_dicts
        start_id = 0
        while start_id < words_count:
            end_id = start_id + 1
            while end_id < words_count and work_data.words[end_id].columns[ID] != '1':
                end_id += 1

            # print(start_id, end_id)

            char_ids = []
            word_ids = []
            first_ids = []
            last_ids = []
            char_pos = 0

            # end_word_id stops on first token of next sentence
            for token in work_data.words[start_id: end_id]:
                token_chars = [
                    labeled_data.lexicon.get_char(char)
                    for char
                    in work_data.characters[token.span.start: token.span.end]
                ]
                char_ids.extend(token_chars)
                char_ids.append(labeled_data.lexicon.get_char(' '))
                first_ids.append(char_pos)
                char_pos += len(token_chars)
                last_ids.append(char_pos - 1)
                # add one for space between words
                char_pos += 1
                word_ids.append(
                    labeled_data.lexicon.get_word(token.columns[FORM])
                )

            _, _, probabilities = self.forward(
                char_ids=torch.LongTensor(char_ids).to(self.device),
                word_ids=torch.LongTensor(word_ids).to(self.device),
                first_ids=torch.LongTensor(last_ids).to(self.device),
                last_ids=torch.LongTensor(first_ids).to(self.device),
            )

            # TODO zip Tensor? make this better
            for tag_id, token in zip(
                    probabilities.argmax(dim=1),
                    work_data.words[start_id: end_id]
            ):
                token.columns[tag_column_id] = labeled_data.tags[tag_name].get_value(tag_id)
                # print(token.columns)

            start_id = end_id

        gold_data = load_conllu_file(path)
        scores = evaluate(gold_data, work_data)

        # TODO maybe unpack this foreign scores
        return scores


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

            # TODO this returns much more than just a number right now. check!
            f1 = self.dev_eval()
            if f1 > best_f1:
                # implement this
                self.lock_best_model()
                best_f1 = f1
