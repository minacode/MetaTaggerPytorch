from Corpora.ud_test_v2_0_conll2017.evaluation_script.conll17_ud_eval import load_conllu_file, ID, FORM
import json
from Lexicon import LabeledData, get_labeled_data_path, Sentence, TAG_NAMES, tag_name_to_column
from typing import List


def new_sentence():
    return Sentence(
        char_ids=[],
        word_ids=[],
        first_ids=[],
        last_ids=[],
        tag_ids={
            tag_name: []
            for tag_name in TAG_NAMES
        }
    )


def get_converted_data_path(dataset, language):
    return f'Datasets/{dataset}/{language}.json'


def load_converted_data(language, dataset):
    with open(f'Datasets/{dataset}/{language}.json') as file:
        return json.load(file)


def create_language_files(dataset: str, language: str, path: str) -> None:
    # sorted list of all words from all sentences
    corpus = load_conllu_file(path)

    labeled_data: LabeledData = LabeledData(
        dataset=dataset,
        language=language,
        labels=TAG_NAMES
    )
    labeled_data.lexicon.add_char(' ')
    sentences: List[Sentence] = []

    char_pos: int = 0
    sentence: Sentence = new_sentence()

    # n_sentences = 0

    for token in corpus.words:
        # check for start of new sentence
        if token.columns[ID] == '1':

            # just for overfitting
            # n_sentences += 1
            # if n_sentences > 1:
            #     break

            # TODO do not add sentence at first iteration, delete workaround lower
            sentence['char_ids'] = sentence['char_ids'][:-1]
            sentences.append(sentence)
            char_pos = 0
            sentence = new_sentence()

        labeled_data.lexicon.add_word(token.columns[FORM])
        for tag_name in TAG_NAMES:
            labeled_data.tags[tag_name].add(
                token.columns[
                    tag_name_to_column(tag_name)
                ]
            )

        # token_chars = labeled_data.lexicon.get_chars(token[FORM])
        token_chars = [
            labeled_data.lexicon.get_char(char)
            for char
            in corpus.characters[token.span.start: token.span.end]
        ]
        sentence['char_ids'].extend(token_chars)
        sentence['char_ids'].append(labeled_data.lexicon.get_char(' '))
        sentence['first_ids'].append(char_pos)
        char_pos += len(token_chars)
        sentence['last_ids'].append(char_pos - 1)
        # add one for space between words
        char_pos += 1

        sentence['word_ids'].append(
            labeled_data.lexicon.get_word(token.columns[FORM])
        )
        for tag_name in TAG_NAMES:
            sentence['tag_ids'][tag_name].append(
                labeled_data.tags[tag_name].get(
                    token.columns[
                        tag_name_to_column(tag_name)
                    ]
                )
            )

    # also add last sentence
    sentence['char_ids'] = sentence['char_ids'][:-1]
    sentences.append(sentence)
    # strip first (always empty) sentence
    sentences = sentences[1:]

    path = get_converted_data_path(
        dataset=dataset,
        language=language
    )
    with open(path, 'w') as file:
        json.dump(sentences, file, indent=4)

    labeled_data.save(
        get_labeled_data_path(
            dataset=dataset,
            language=language
        )
    )


# TODO save labels for upostag and xpostag
def convert_data():
    with open('data.json', 'r') as file:
        data = json.load(file)
    for dataset in data:
        for language in data[dataset]:
            train_path = data[dataset][language]['train']
            create_language_files(
                dataset=dataset,
                language=language,
                path=train_path
            )


def print_converted_data_test():
    labeled_data = LabeledData.load('Dictionaries/conll17/de.json')
    sentences = load_converted_data(
        dataset='conll17',
        language='de'
    )
    for sentence in sentences:
        print(
            ' '.join([
                labeled_data.lexicon.get_word_by_id(word_id)
                for word_id
                in sentence['words']
            ])
        )


if __name__ == "__main__":
    convert_data()
