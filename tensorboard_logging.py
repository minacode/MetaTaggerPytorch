# meta functions
def log_log_histogram(writer, steps, name, tensor):
    writer.add_histogram(
        name,
        (tensor.abs() + 1e-8).log(),
        steps
    )


# main logging functions
def log_epoch(writer, epoch, f1, best_f1, best_epoch):
    writer.add_scalar('epochs/f1', f1, epoch)
    writer.add_scalar('epochs/best_f1', best_f1, epoch)
    writer.add_scalar('epochs/best_epoch', best_epoch, epoch)


def log_chars(writer, model, steps, one_loss, losses_out, combined_losses):
    log_char_net(
        writer=writer,
        model=model,
        steps=steps
    )
    log_losses(
        writer=writer,
        steps=steps,
        name='char',
        one_loss=one_loss,
        losses_out=losses_out,
        combined_losses=combined_losses
    )


def log_words(writer, model, steps, one_loss, losses_out, combined_losses):
    log_word_net(
        writer=writer,
        model=model,
        steps=steps
    )
    log_losses(
        writer=writer,
        steps=steps,
        name='word',
        one_loss=one_loss,
        losses_out=losses_out,
        combined_losses=combined_losses
    )


def log_meta(writer, model, steps, one_loss, losses_out, combined_losses):
    log_meta_net(
        writer=writer,
        model=model,
        steps=steps
    )
    log_losses(
        writer=writer,
        steps=steps,
        name='meta',
        one_loss=one_loss,
        losses_out=losses_out,
        combined_losses=combined_losses
    )


# subfunctions
def log_char_net(writer, model, steps):
    log_log_histogram(
        writer=writer,
        steps=steps,
        name='grads/char_embedding',
        tensor=model.char_embedding.weight.grad
    )
    log_log_histogram(
        writer=writer,
        steps=steps,
        name='weights/char_embedding',
        tensor=model.char_embedding.weight,
    )
    model.char_core.log_tensorboard(
        writer=writer,
        name='char_core/',
        iteration_counter=steps
    )
    model.char_classifier.log_tensorboard(
        writer=writer,
        name='char_classifier/',
        iteration_counter=steps
    )


def log_word_net(writer, model, steps):
    log_log_histogram(
        writer=writer,
        steps=steps,
        name='grads/word_embedding',
        tensor=model.word_embedding.weight.grad,
    )
    log_log_histogram(
        writer=writer,
        steps=steps,
        name='weights/word_embedding',
        tensor=model.word_embedding.weight,
    )
    model.word_core.log_tensorboard(
        writer=writer,
        name='word_core/',
        iteration_counter=steps
    )
    model.word_classifier.log_tensorboard(
        writer=writer,
        name='word_classifier/',
        iteration_counter=steps
    )


def log_meta_net(writer, model, steps):
    model.meta_core.log_tensorboard(
        writer=writer,
        name='meta_core/',
        iteration_counter=steps
    )
    model.meta_classifier.log_tensorboard(
        writer=writer,
        name='meta_classifier/',
        iteration_counter=steps
    )


def log_probabilities(writer, steps, probs):
    for name in probs:
        writer.add_histogram(f'probabilities/{name}', probs[name], steps)


def log_losses(writer, steps, name, one_loss, losses_out, combined_losses):
    writer.add_scalar(f'losses/one/{name}', one_loss[name], steps)
    writer.add_scalar(f'losses/losses_out/{name}', losses_out[name], steps)
    writer.add_scalar(f'losses/combined/{name}', combined_losses[name], steps)


def log_learning_rate(writer, steps, decays):
    for name, decay in decays.items():
        writer.add_scalar(
            f'lr/{name}',
            decay.get_lr(),
            steps
        )


def log_embeddings(writer, model, steps, word_list, char_list):
    print('save embeddings')
    writer.add_embedding(
        model.word_embedding.weight,
        global_step=steps,
        tag=f'word_embeddings{steps}',
        metadata=word_list
    )
    writer.add_embedding(
        model.char_embedding.weight,
        global_step=steps,
        tag=f'char_embeddings{steps}',
        metadata=char_list
    )


def log_training(writer, steps, model, n_sentences, one_loss, losses_out, combined_losses, probs, word_list, char_list):
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
        log_embeddings(
            writer=writer,
            model=model,
            steps=steps,
            word_list=word_list,
            char_list=char_list
        )
