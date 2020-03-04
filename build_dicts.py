from typing import List, Dict, Union, Any

from conllu import parse
from Corpora.ud_test_v2_0_conll2017.evaluation_script.conll17_ud_eval import load_conllu_file
from Lexicon import LabeledData, get_labeled_data_path, Sentence, TAG_NAMES, tag_name_to_column
import json
import os.path


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


ID = 0
FORM = 1


def get_converted_data_path(dataset, language):
    return f'Datasets/{dataset}/{language}.json'


def generate_enumerations():
    with open('data.json', 'r') as file:
        data = json.load(file)
    for dataset in data:
        for language in data[dataset]['languages']:
            path = get_labeled_data_path(
                dataset=dataset,
                language=language
            )
            if os.path.isfile(path):
                print(f'skipped creating dictionary for {dataset}: {language}')
            else:
                labeled_data = LabeledData.create_from_language(
                    language=language,
                    dataset=dataset
                )

                # add space manually because it is not observed in the words forming the sentences
                labeled_data.lexicon.add_char(' ')

                labeled_data.save(
                    get_labeled_data_path(
                        dataset=dataset,
                        language=language
                    )
                )


def load_converted_data(language, dataset):
    with open(f'Datasets/{dataset}/{language}.json') as file:
        return json.load(file)


def create_language_files(dataset: str, language: str, data: Dict[str, Any]) -> None:
    base_path: str = data[dataset]['base_path']
    train_directory: str = data[dataset]['train']
    file_name: str = data[dataset]['file_name'].replace('<lang>', language)
    path: str = f'{base_path}/{train_directory}/{file_name}'
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

    for token in corpus.words:
        # check for start of new sentence
        if token.columns[ID] == '1':
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


def create_language_files_external(dataset: str, language: str, data):
    # print(f'Convert data for {language}…', end='')
    labeled_data: LabeledData = LabeledData.load(
        get_labeled_data_path(
            dataset=dataset,
            language=language
        )
    )

    base_path: str = data[dataset]['base_path']
    train_directory: str = data[dataset]['train']
    file_name: str = data[dataset]['file_name'].replace('<lang>', language)
    path: str = f'{base_path}/{train_directory}/{file_name}'
    with open(path, 'r') as file:
        parsed = parse(file.read())

    # nearly , but tag_ids is more complicated
    sentences: List[Dict[str, Union[List[int], Dict[str, List[int]]]]] = []
    for sentence in parsed:
        char_ids: List[int] = []
        word_ids: List[int] = []
        tag_ids: Dict[str, List[int]] = {
            label: []
            for label in labeled_data.labels
        }
        first_ids: List[int] = [0]
        last_ids: List[int] = []
        char_pos: int = 0
        for token in sentence:
            token_chars = labeled_data.lexicon.get_chars(token['form'])
            char_ids.extend(token_chars)
            char_ids.append(labeled_data.lexicon.get_char(' '))
            char_pos += len(token_chars) + 1
            last_ids.append(char_pos - 2)
            first_ids.append(char_pos)

            word_ids.append(
                labeled_data.lexicon.get_word(token['form'])
            )
            for label in labeled_data.labels:
                tag_ids[label].append(
                    labeled_data.tags[label].get(token[label])
                )

        sentences.append({
            'chars': char_ids,
            'words': word_ids,
            'firsts': first_ids,
            'lasts': last_ids,
            'tags': tag_ids
        })

    path = get_converted_data_path(
        dataset=dataset,
        language=language
    )
    with open(path, 'w') as file:
        json.dump(sentences, file, indent=4)
    del labeled_data, sentences, char_ids, word_ids, tag_ids, first_ids, last_ids
    print(f'Convert data for {language} done.')


# TODO save labels for upostag and xpostag
def convert_data():
    with open('data.json', 'r') as file:
        data = json.load(file)
    for dataset in data:
        for language in data[dataset]['languages']:
            create_language_files(
                dataset=dataset,
                language=language,
                data=data
            )


def test_print_converted_data():
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
    # generate_enumerations()
    convert_data()
