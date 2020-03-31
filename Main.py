from build_dicts import load_converted_data, convert_data
from evaluation import evaluate_model
from Lexicon import LabeledData, get_labeled_data_path
import json
from LSTMModel import LSTMModel
import sys
import torch
from time import asctime
from train import train
from itertools import chain


def run_complete_net(dataset, language, tag_name, epochs, embedding_dim, debug=False):
    with open('data.json', 'r') as file:
        data = json.load(file)

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


def evaluate(dataset, language, tag_name, model_path, data_path):
    labels = LabeledData.load(
        get_labeled_data_path(
            dataset=dataset,
            language=language
        )
    )
    model = torch.load(model_path)
    scores = evaluate_model(
        model=model,
        tag_name=tag_name,
        path=data_path,
        labeled_data=labels
    )
    for name in scores:
        print(name, scores[name].f1)


if __name__ == "__main__":
    args = sys.argv
    n_args = len(args)
    if n_args >= 2:
        command = args[1]
        if command == 'train' and n_args == 7:
            dataset = args[2]
            language = args[3]
            tag_name = args[4]
            epochs = int(args[5])
            embedding_dim = int(args[6])

            run_complete_net(
                dataset=dataset,
                language=language,
                tag_name=tag_name,
                epochs=epochs,
                embedding_dim=embedding_dim,
                debug=False
            )
        elif command == 'convert':
            convert_data()
        elif command == 'evaluate' and n_args == 7:
            dataset = args[2]
            language = args[3]
            tag_name = args[4]
            model_path = args[5]
            data_path = args[6]
            evaluate(
                dataset=dataset,
                language=language,
                tag_name=tag_name,
                model_path=model_path,
                data_path=data_path
            )
        elif command == 'help':
            print('options:')
            print('train dataset language tag_name epochs embedding_dim')
            print('-> train new model with parameters')
            print('convert')
            print('-> convert datasets in data.json')
            print('eval dataset language tag_name model_path data_path')
            print('-> evaluate model saved in model_path with conllu file in data_path')
