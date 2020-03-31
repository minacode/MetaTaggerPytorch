from evaluation import evaluate_model
from pathlib import Path
from random import shuffle
from tensorboardX import SummaryWriter
from torch import tensor, long, save
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.sparse_adam import SparseAdam
from tensorboard_logging import log_epoch, log_chars, log_words, log_meta
import unicodedata


def get_losses(loss_mode):
    if loss_mode == 'mse':
        char_loss = MSELoss(reduction='mean')
        word_loss = MSELoss(reduction='mean')
        meta_loss = MSELoss(reduction='mean')
    elif loss_mode == 'ce':
        char_loss = CrossEntropyLoss()
        word_loss = CrossEntropyLoss()
        meta_loss = CrossEntropyLoss()
    else:
        raise Exception(f'invalid loss instantiation: {loss_mode}')
    return {
        'char': char_loss,
        'word': word_loss,
        'meta': meta_loss
    }


def get_optimizers(model, optimizer_type, optimizer_args):
    if optimizer_type == 'adam':
        cls = Adam
        parameters = ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
    elif optimizer_type == 'sparse_adam':
        cls = SparseAdam
        parameters = ['lr', 'betas', 'eps']
    elif optimizer_type == 'sgd':
        cls = SGD
        parameters = ['lr']
    else:
        raise Exception(f'unknown optimizer type: {optimizer_type}')

    filtered_args = {
        arg: optimizer_args[arg]
        for arg in optimizer_args
        if arg in parameters
    }

    char_optimizer = cls(model.get_char_params(), **filtered_args)
    word_optimizer = cls(model.get_word_params(), **filtered_args)
    meta_optimizer = cls(model.get_meta_params(), **filtered_args)
    return {
        'char': char_optimizer,
        'word': word_optimizer,
        'meta': meta_optimizer
    }


def get_base_tensors(sentence, model, tag_name, n_tags, loss_mode):
    # set base tensors
    chars = tensor(data=sentence['char_ids'], dtype=long, device=model.device)
    words = tensor(sentence['word_ids'], dtype=long, device=model.device)
    # targets = torch.LongTensor(sentence['tag_ids'][tag_name]).to(self.device)
    if loss_mode == 'mse':
        targets = [
            [0 for _ in range(n_tags)]
            for _ in range(len(sentence['tag_ids'][tag_name]))
        ]
        for n_tag, tag_id in enumerate(sentence['tag_ids'][tag_name]):
            targets[n_tag][tag_id] = 1
        targets = tensor(data=targets, device=model.device)
    elif loss_mode == 'ce':
        targets = tensor(data=sentence['tag_ids'][tag_name], dtype=long, device=model.device)
    else:
        raise Exception(f'invalid loss mode while creating base tensors: {loss_mode}')
    firsts = tensor(data=sentence['first_ids'], dtype=long, device=model.device)
    lasts = tensor(data=sentence['last_ids'], dtype=long, device=model.device)
    return chars, words, targets, firsts, lasts


def evaluate_probs(probs, targets, loss_mode, writer, steps):
    if loss_mode == 'ce':
        target_predictions = targets
    elif loss_mode == 'mse':
        target_predictions = targets.argmax(dim=1)
    else:
        raise Exception(f'unknown loss mode: {loss_mode}')

    for a, b in zip(
            probs['meta'].argmax(dim=1),
            target_predictions
    ):
        print(f'{a} {b}', end='|')
    print()

    predictions = {
        name: probs[name].argmax(dim=1)
        for name in probs
    }

    wrongs = {
        name: 0
        for name in predictions
    }
    for name in predictions:
        for i in range(len(predictions[name])):
            if predictions[name][i] != target_predictions[i]:
                wrongs[name] += 1
    # normalise wrongs from 0 to 1
    # 0.. all words wrong labeled
    # 1.. all words correct labeled
    wrongs = {
        name: wrongs[name] / len(target_predictions)
        for name in wrongs
    }

    print(wrongs)
    for name in wrongs:
        writer.add_scalar(f'wrongs/{name}', wrongs[name], steps)

    # print(model.word_core.linear.weight.round())
    # print(model.word_core.linear.bias.round())


def get_losses_for_training(probs, targets, losses, train_by):
    one_loss = {
        name: (1 - probs[name].sum(dim=1)).abs().sum(dim=0)
        for name in probs
    }
    names_in_probs_and_losses = set(probs.keys()).intersection(losses.keys())
    losses_out = {
        name: losses[name](probs[name], targets)
        for name in names_in_probs_and_losses
    }
    combined_losses = {
        name: losses_out[name] + one_loss[name]
        for name in losses_out
    }

    if train_by == 'combined':
        losses_for_training = combined_losses
    elif train_by == 'out':
        losses_for_training = losses_out
    elif train_by == 'one':
        losses_for_training = one_loss
    else:
        raise Exception(f'unknown training mode: {train_by}')

    return losses_for_training, one_loss, losses_out, combined_losses


def get_word_list(labeled_data):
    word_list = labeled_data.lexicon._words.to_dict()['elements']
    word_list_unk = [
        repr(repr(word)) for word in
        ['unknown'] + word_list
    ]
    return word_list_unk


def get_char_list(labeled_data):
    char_list = labeled_data.lexicon._chars.to_dict()['elements']
    char_list_unk = ['unknown']
    for char in char_list:
        # print(char, unicodedata.name(char))
        char_list_unk.append(
            unicodedata.name(char)
        )
    return char_list_unk


def train_char_net(model, sentence, labeled_data, losses, optimizers, tag_name, loss_mode, train_by, steps, writer):
    n_tags = labeled_data.get_n_tags(tag_name=tag_name)

    char_ids, _, targets, first_ids, last_ids = get_base_tensors(
        sentence=sentence,
        model=model,
        tag_name=tag_name,
        n_tags=n_tags,
        loss_mode=loss_mode
    )

    probs = {
        'char': model.get_char_probabilities((char_ids, first_ids, last_ids))
    }

    losses_for_training, one_loss, losses_out, combined_losses = get_losses_for_training(
        probs=probs,
        targets=targets,
        losses=losses,
        train_by=train_by
    )
    losses_for_training['char'].backward()
    if not steps % 10:
        log_chars(
            writer=writer,
            model=model,
            steps=steps,
            one_loss=one_loss,
            losses_out=losses_out,
            combined_losses=combined_losses
        )
    optimizers['char'].step()
    optimizers['char'].zero_grad()


def train_word_net(model, sentence, labeled_data, losses, optimizers, tag_name, loss_mode, train_by, steps, writer):
    n_tags = labeled_data.get_n_tags(tag_name=tag_name)

    _, word_ids, targets, _, _ = get_base_tensors(
        sentence=sentence,
        model=model,
        tag_name=tag_name,
        n_tags=n_tags,
        loss_mode=loss_mode
    )

    probs = {
        'word': model.get_word_probabilities(word_ids)
    }

    losses_for_training, one_loss, losses_out, combined_losses = get_losses_for_training(
        probs=probs,
        targets=targets,
        losses=losses,
        train_by=train_by
    )
    losses_for_training['word'].backward()
    if not steps % 10:
        log_words(
            writer=writer,
            model=model,
            steps=steps,
            one_loss=one_loss,
            losses_out=losses_out,
            combined_losses=combined_losses
        )
    optimizers['word'].step()
    optimizers['word'].zero_grad()


def train_meta_net(model, sentence, labeled_data, losses, optimizers, tag_name, loss_mode, train_by, steps, writer):
    n_tags = labeled_data.get_n_tags(tag_name=tag_name)

    char_ids, word_ids, targets, first_ids, last_ids = get_base_tensors(
        sentence=sentence,
        model=model,
        tag_name=tag_name,
        n_tags=n_tags,
        loss_mode=loss_mode
    )

    probs = {
        'meta': model.get_meta_probabilities((char_ids, word_ids, first_ids, last_ids))
    }

    losses_for_training, one_loss, losses_out, combined_losses = get_losses_for_training(
        probs=probs,
        targets=targets,
        losses=losses,
        train_by=train_by
    )
    losses_for_training['meta'].backward()
    if not steps % 10:
        log_meta(
            writer=writer,
            model=model,
            steps=steps,
            one_loss=one_loss,
            losses_out=losses_out,
            combined_losses=combined_losses
        )
    optimizers['meta'].step()
    for name in optimizers:
        optimizers[name].zero_grad()


# TODO finish this
def train(dataset, language, tag_name, model, labeled_data, sentences, epochs, test_data_path, timestamp):
    loss_mode = 'ce'
    train_by = 'out'
    optimizer_type = 'sgd'

    score_names = {
        'POS': 'UPOS',
        'XPOS': 'XPOS',
        'FEATURE': 'Feats'
    }

    optimizer_args = {
        'lr': 0.002,  # TODO test other values
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.999994
    }

    n_tags = labeled_data.get_n_tags(tag_name=tag_name)
    word_list = get_word_list(labeled_data=labeled_data)
    char_list = get_char_list(labeled_data=labeled_data)

    losses = get_losses(loss_mode=loss_mode)

    optimizers = get_optimizers(model, optimizer_type, optimizer_args)
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    writer = SummaryWriter(comment=f'_{dataset}_{language}_{tag_name}')

    best_f1 = 0
    best_epoch = 0

    steps = 0

    for n_epoch in range(epochs):
        print(f'starting epoch {n_epoch}')
        shuffle(sentences)
        base_step = steps
        model.train()
        for sentence in sentences:
            train_char_net(
                model=model,
                sentence=sentence,
                labeled_data=labeled_data,
                losses=losses,
                optimizers=optimizers,
                tag_name=tag_name,
                loss_mode=loss_mode,
                train_by=train_by,
                steps=steps,
                writer=writer
            )
            steps += 1
        steps = base_step

        for sentence in sentences:
            train_word_net(
                model=model,
                sentence=sentence,
                labeled_data=labeled_data,
                losses=losses,
                optimizers=optimizers,
                tag_name=tag_name,
                loss_mode=loss_mode,
                train_by=train_by,
                steps=steps,
                writer=writer
            )
            steps += 1
        steps = base_step

        for sentence in sentences:
            train_meta_net(
                model=model,
                sentence=sentence,
                labeled_data=labeled_data,
                losses=losses,
                optimizers=optimizers,
                tag_name=tag_name,
                loss_mode=loss_mode,
                train_by=train_by,
                steps=steps,
                writer=writer
            )
            steps += 1

        '''
        if not steps % 100:
            evaluate_probs(
                probs=probs,
                targets=targets,
                loss_mode=loss_mode,
                writer=writer,
                steps=steps
            )
        '''

        # save model
        model_path = Path(f'Models/{dataset}/{language}/{tag_name}/{timestamp}')
        model_path.mkdir(
            parents=True,
            exist_ok=True
        )
        file_name = Path(f'{n_epoch}.model')
        save(model, model_path.joinpath(file_name))

        # evaluate model
        scores = evaluate_model(
            model=model,
            tag_name=tag_name,
            path=test_data_path,
            labeled_data=labeled_data
        )
        score_name = score_names[tag_name]
        f1 = scores[score_name].f1
        if f1 > best_f1:
            best_epoch = n_epoch
            best_f1 = f1

        log_epoch(
            writer=writer,
            epoch=n_epoch,
            f1=f1,
            best_f1=best_f1,
            best_epoch=best_epoch
        )

    writer.close()
