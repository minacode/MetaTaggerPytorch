from random import shuffle
from tensorboardX import SummaryWriter
from torch import tensor, long
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.sparse_adam import  SparseAdam
from tensorboard_logging import log_losses, log_probabilities


def log(writer, steps, model, n_sentences, one_loss, losses_out, combined_losses, probs, word_list, char_list):
    if steps % 10:
        return
    log_losses(
        writer=writer,
        steps=steps,
        n_sentences=n_sentences,
        one_loss=one_loss,
        losses_out=losses_out,
        combined_losses=combined_losses
    )

    if not steps % 10:
        log_probabilities(
            writer=writer,
            steps=steps,
            probs=probs
        )

    if not steps % 1000:
        model.log_embeddings(
            writer=writer,
            steps=steps,
            word_list=word_list,
            char_list=char_list
        )


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
        for name in losses
    }
    losses_out = {
        name: losses[name](probs[name], targets)
        for name in losses
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


# TODO finish this
def train(dataset, language, model, sentences, epochs, n_tags, tag_name, word_list, char_list):
    loss_mode = 'ce'
    train_by = 'out'
    optimizer_type = 'sgd'

    optimizer_args = {
        'lr': 0.002,  # TODO test other values
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.999994
    }

    losses = get_losses(loss_mode=loss_mode)

    optimizers = get_optimizers(model, optimizer_type, optimizer_args)
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    model.train()

    writer = SummaryWriter(comment=f'_{dataset}_{language}')

    n_sentences = len(sentences)
    steps = 0

    for n_epoch in range(epochs):
        # print(f'starting epoch {n_epoch}')
        shuffle(sentences)
        for sentence in sentences:
            chars, words, targets, firsts, lasts = get_base_tensors(
                sentence=sentence,
                model=model,
                tag_name=tag_name,
                n_tags=n_tags,
                loss_mode=loss_mode
            )

            cp, wp, mp = model([chars, words, firsts, lasts])
            probs = {
                'char': cp,
                'word': wp,
                'meta': mp
            }

            if not steps % 100:
                evaluate_probs(
                    probs=probs,
                    targets=targets,
                    loss_mode=loss_mode,
                    writer=writer,
                    steps=steps
                )

            losses_for_training, one_loss, losses_out, combined_losses = get_losses_for_training(
                probs=probs,
                targets=targets,
                losses=losses,
                train_by=train_by
            )

            losses_for_training['char'].backward(retain_graph=True)
            if not steps % 10:
                model.log_char_net(
                    writer=writer,
                    steps=steps
                )
            optimizers['char'].step()
            optimizers['char'].zero_grad()

            losses_for_training['word'].backward(retain_graph=True)
            if not steps % 10:
                model.log_char_net(
                    writer=writer,
                    steps=steps
                )
            optimizers['word'].step()
            optimizers['word'].zero_grad()

            losses_for_training['meta'].backward()
            if not steps % 10:
                model.log_char_net(
                    writer=writer,
                    steps=steps
                )
            optimizers['meta'].step()
            optimizers['meta'].zero_grad()

            log(
                writer=writer,
                steps=steps,
                model=model,
                n_sentences=n_sentences,
                one_loss=one_loss,
                losses_out=losses_out,
                combined_losses=combined_losses,
                probs=probs,
                word_list=word_list,
                char_list=char_list,
            )

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            steps += 1

    writer.close()
