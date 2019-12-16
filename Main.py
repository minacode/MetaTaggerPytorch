from build_dicts import load_converted_data
from Dictionary import LabeledData
import torch
from LSTMModel import LSTMModel


def test_complete_net():
    labels = LabeledData.from_language('de')
    sentences = load_converted_data(
        language='de',
        dataset='conll17'
    )

    embedding_dim = 10

    model = LSTMModel(
        n_chars=labels.dictionary.n_chars(),
        n_words=labels.dictionary.n_words(),
        n_tags=len(labels.tags),
        embedding_dim=embedding_dim
    )
    model.run_training(
        sentences=sentences,
        epochs=10
    )
    torch.save(model.get_state_dicts(), 'Models/1')


if __name__ == "__main__":
    test_complete_net()
