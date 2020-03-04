from build_dicts import load_converted_data, convert_data
from Lexicon import LabeledData, get_labeled_data_path
import json
import torch
from LSTMModel import LSTMModel


def run_complete_net(debug=False):
    dataset = 'conll17'
    tag_name = 'XPOS'
    epochs = 1

    with open('data.json', 'r') as file:
        data = json.load(file)
    
    for language in data[dataset]['languages']:
        labels = LabeledData.load(
            get_labeled_data_path(
                dataset=dataset,
                language=language
            )
        )
        sentences = load_converted_data(
            language=language,
            dataset=dataset
        )

        embedding_dim = 10

        model = LSTMModel(
            n_chars=labels.lexicon.n_chars(),
            n_words=labels.lexicon.n_words(),
            n_tags=len(labels.tags[tag_name]),
            embedding_dim=embedding_dim,
            cuda=False,
            debug=debug
        )

        model.run_training(
            sentences=sentences,
            epochs=epochs,
            tag_name=tag_name
        )
        torch.save(
            model.get_state_dicts(
                language=language,
                dataset=dataset
            ), 
            f'Models/conll17/{language}'
        )


if __name__ == "__main__":
    convert_data()
    run_complete_net(debug=False)
