from build_dicts import load_converted_data, convert_data
from Lexicon import LabeledData, get_labeled_data_path
import json
import torch
from LSTMModel import LSTMModel
from time import asctime
from train import train
from itertools import chain


def run_complete_net(debug=False):
    dataset = 'conll17'
    tag_name = 'XPOS'
    epochs = 20

    with open('data.json', 'r') as file:
        data = json.load(file)

    for language in data[dataset]:
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

        n_tags = labels.get_n_tags(tag_name=tag_name)
        embedding_dim = 400

        model = LSTMModel(
            n_chars=labels.lexicon.n_chars(),
            n_words=labels.lexicon.n_words(),
            n_tags=n_tags,
            embedding_dim=embedding_dim,
            cuda=True,
            debug=debug,
            residual=False
        )

        # check if the functions handle every parameter
        model_parameters = set(model.parameters())
        single_parameters = set(chain(
            model.get_char_params(),
            model.get_word_params(),
            model.get_meta_params()
        ))
        assert model_parameters == single_parameters

        model.initialise()

        timestamp = asctime().replace(' ', '_')
        dev_path = data[dataset][language]['dev']

        train(
            dataset=dataset,
            language=language,
            tag_name=tag_name,
            model=model,
            labeled_data=labels,
            sentences=sentences,
            epochs=epochs,
            test_data_path=dev_path,
            timestamp=timestamp
        )


if __name__ == "__main__":
    # convert_data()
    run_complete_net(debug=False)
