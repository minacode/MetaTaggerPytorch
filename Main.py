from build_dicts import load_converted_data, convert_data
from Lexicon import LabeledData, get_labeled_data_path
import json
import torch
from LSTMModel import LSTMModel
from time import asctime
from train import train
import unicodedata
from itertools import chain


def run_complete_net(debug=False):
    dataset = 'conll17'
    tag_name = 'POS'
    epochs = 10

    with open('data.json', 'r') as file:
        data = json.load(file)
    
    for language in ['de']:  # data[dataset]['languages']:
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

        n_tags = len(labels.tags[tag_name])
        # TODO hack
        embedding_dim = 50

        model = LSTMModel(
            n_chars=labels.lexicon.n_chars(),
            n_words=labels.lexicon.n_words(),
            n_tags=n_tags,
            embedding_dim=embedding_dim,
            cuda=True,
            debug=debug,
            residual=True
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

        word_list = labels.lexicon._words.to_dict()['elements']
        # TODO rename blubb to unknown
        word_list_unk = [
            repr(repr(word)) for word in
            ['unknown'] + word_list
        ]

        char_list = labels.lexicon._chars.to_dict()['elements']
        char_list_unk = ['unknown']
        for char in char_list:
            # print(char, unicodedata.name(char))
            char_list_unk.append(
                unicodedata.name(char)
            )

        torch.autograd.set_detect_anomaly(True)

        train(
            dataset=dataset,
            language=language,
            model=model,
            sentences=sentences,
            epochs=epochs,
            n_tags=n_tags,
            tag_name=tag_name,
            word_list=word_list_unk,
            char_list=char_list_unk
        )

        timestamp = asctime().replace(' ', '_')
        torch.save(model, f'Models/{dataset}/{language}_{timestamp}')

        print('training finished, starting evaluation')

        if False:
            scores = model.dev_eval(
                tag_name=tag_name,
                path='Corpora/ud_test_v2_0_conll2017/gold/conll17-ud-test-2017-05-09/de.conllu',
                labeled_data=labels
            )
            for category in scores:
                print(category)
                print(scores[category].precision)
                print(scores[category].recall)
                print(scores[category].f1)

        # TODO save is broken, optimizers no longer in lstm-model
        '''
        torch.save(
            model.get_state_dicts(
                language=language,
                dataset=dataset
            ), 
            f'Models/conll17/{language}'
        )
        '''


if __name__ == "__main__":
    # convert_data()
    run_complete_net(debug=False)
