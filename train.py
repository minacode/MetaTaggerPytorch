from random import shuffle
from tensorboardX import SummaryWriter
from torch import LongTensor, Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def log_probabilities(writer, steps, probs):
    for name, prob in probs.items():
        writer.add_histogram(f'probabilities/{name}', probs[name], steps)


def log_losses(writer, steps, n_sentences, losses_out):
    for name in losses_out:
        writer.add_scalar(f'losses/{name}', losses_out[name], steps)
    if not steps % 100:
        print(
            f'{steps}/{n_sentences}\t'
            f'{losses_out["char"].item()}\t'
            f'{losses_out["word"].item()}\t'
            f'{losses_out["meta"].item()}'
        )


def log_learning_rate(writer, steps, decays):
    for name, decay in decays.items():
        writer.add_scalar(
            'name/lr',
            decay.get_lr(),
            steps
        )


def log(writer, steps, model, n_sentences, losses_out, probs, word_list):
    log_losses(
        writer=writer,
        steps=steps,
        n_sentences=n_sentences,
        losses_out=losses_out
    )

    if not steps % 10:
        log_probabilities(
            writer=writer,
            steps=steps,
            probs=probs
        )
        model.log_grads(
            writer=writer,
            steps=steps
        )

    if not steps % 1000:
        model.log_embeddings(
            writer=writer,
            steps=steps,
            word_list=word_list
        )


def get_losses():
    char_loss = MSELoss(reduction='sum')
    word_loss = MSELoss(reduction='sum')
    meta_loss = MSELoss(reduction='sum')
    return {
        'char': char_loss,
        'word': word_loss,
        'meta': meta_loss
    }


def get_optimizers(model, optimizer_args):
    char_optimizer = Adam(model.get_char_params(), **optimizer_args)
    word_optimizer = Adam(model.get_word_params(), **optimizer_args)
    meta_optimizer = Adam(model.get_meta_params(), **optimizer_args)
    return {
        'char': char_optimizer,
        'word': word_optimizer,
        'meta': meta_optimizer
    }


def get_decays(gamma, optimizers):
    return {
        name: ExponentialLR(
            optimizer=optimizers[name],
            gamma=gamma
        )
        for name in optimizers
    }


def get_base_tensors(sentence, model, tag_name, n_tags):
    # set base tensors
    chars = LongTensor(sentence['char_ids']).to(model.device)
    words = LongTensor(sentence['word_ids']).to(model.device)
    # targets = torch.LongTensor(sentence['tag_ids'][tag_name]).to(self.device)
    targets = [
        [0 for _ in range(n_tags)]
        for _ in range(len(sentence['tag_ids'][tag_name]))
    ]
    for n_tag, tag_id in enumerate(sentence['tag_ids'][tag_name]):
        targets[n_tag][tag_id] = 1
    targets = Tensor(targets).to(model.device)
    firsts = LongTensor(sentence['first_ids']).to(model.device)
    lasts = LongTensor(sentence['last_ids']).to(model.device)
    return chars, words, targets, firsts, lasts


# TODO finish this
def train(model, sentences, epochs, n_tags, tag_name, word_list):
    losses = get_losses()

    optimizer_args = {
        'lr': 0.002,  # TODO test other values
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.999994
    }
    optimizers = get_optimizers(model, optimizer_args)

    gamma = 0.999
    decays = get_decays(
        gamma=gamma,
        optimizers=optimizers
    )

    # set model to train mode
    model.train()
    writer = SummaryWriter()
    n_sentences = len(sentences)
    steps = 0

    for n_epoch in range(epochs):
        print(f'starting epoch {n_epoch}')
        shuffle(sentences)
        for sentence in sentences:

            # reset gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            chars, words, targets, firsts, lasts = get_base_tensors(
                sentence=sentence,
                model=model,
                tag_name=tag_name,
                n_tags=n_tags
            )

            cp, wp, mp = model(
                char_ids=chars,
                word_ids=words,
                first_ids=firsts,
                last_ids=lasts
            )
            probs = {
                'char': cp,
                'word': wp,
                'meta': mp
            }

            # targets = targets.permute(1, 0)

            one_loss = {
                name: (1 - probs[name].sum(dim=1)).sum(dim=0)
                for name in losses
            }

            losses_out = {
                name: losses[name](probs[name], targets) + one_loss[name]
                for name in losses
            }

            losses_out['char'].backward(retain_graph=True)
            losses_out['word'].backward(retain_graph=True)
            losses_out['meta'].backward()

            log(
                writer=writer,
                steps=steps,
                model=model,
                n_sentences=n_sentences,
                losses_out=losses_out,
                probs=probs,
                word_list=word_list
            )

            for optimizer in optimizers.values():
                optimizer.step()

            for decay in decays.values():
                decay.step()

            steps += 1

    writer.close()
