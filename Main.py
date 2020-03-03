from build_dicts import load_converted_data
from Dictionary import LabeledData
import json
import torch
from LSTMModel import LSTMModel


def test_complete_net():
    dataset = 'conll17'

    with open('data.json', 'r') as file:
        data = json.load(file)
    
    for language in data[dataset]['languages']:
        labels = LabeledData.from_language(language)
        sentences = load_converted_data(
            language=language,
            dataset=dataset
        )

        embedding_dim = 400

        model = LSTMModel(
            n_chars=labels.dictionary.n_chars(),
            n_words=labels.dictionary.n_words(),
            n_tags=len(labels.tags),
            embedding_dim=embedding_dim,
            cuda=True
        )

        model.run_training(
            sentences=sentences,
            epochs=20
        )
        torch.save(
            model.get_state_dicts(
                language=language,
                dataset=dataset
            ), 
            f'Models/conll17/{language}'
        )


if __name__ == "__main__":
    test_complete_net()
