def log_probabilities(writer, steps, probs):
    for name, prob in probs.items():
        writer.add_histogram(f'probabilities/{name}', probs[name], steps)


def log_losses(writer, steps, n_sentences, one_loss, losses_out, combined_losses):
    for name in losses_out:
        writer.add_scalar(f'losses/one/{name}', one_loss[name], steps)
        writer.add_scalar(f'losses/losses_out/{name}', losses_out[name], steps)
        writer.add_scalar(f'losses/combined/{name}', combined_losses[name], steps)
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
            f'lr/{name}',
            decay.get_lr(),
            steps
        )


def log_log_histogram(writer, steps, name, tensor):
    writer.add_histogram(
        name,
        (tensor.abs() + 1e-8).log(),
        steps
    )


def log_epoch(writer, epoch, f1, best_f1, best_epoch):
    writer.add_scalar('f1', f1, epoch)
    writer.add_scalar('best_f1', best_f1, epoch)
    writer.add_scalar('best_epoch', best_epoch, epoch)

