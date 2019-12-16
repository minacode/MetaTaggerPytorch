from conllu import parse, parse_incr
from Dictionary import LabeledData
from itertools import accumulate
import json


def generate_enumerations():
    with open('data.json', 'r') as file:
        data = json.load(file)
    for dataset in data:
        for language in data[dataset]['languages']:
            LabeledData.from_language(
                language=language,
                dataset=dataset
            ).save()


def load_converted_data(language, dataset):
    with open(f'Datasets/{dataset}/{language}.json') as file:
        return json.load(file)


def convert_data():
    with open('data.json', 'r') as file:
        data = json.load(file)
    for dataset in data:
        for language in data[dataset]['languages']:
            print(f'Convert data for {language}…', end='')
            labels = LabeledData.from_language(
                language=language,
                dataset=dataset
            )

            base_path = data[dataset]['base_path']
            train_directory = data[dataset]['train']
            file_name = data[dataset]['file_name'].replace('<lang>', language)
            path = f'{base_path}/{train_directory}/{file_name}'
            with open(path, 'r') as file:
                lines = file.read()

            parsed = parse(lines)

            sentences = []
            for sentence in parsed:
                chars = sum(
                    [
                        labels.dictionary.get_chars(token['form'])
                        for token in sentence
                    ],
                    []
                )
                words = [labels.dictionary.get_word(token['form']) for token in sentence]
                tags = [labels.tags[token['upostag']] for token in sentence]

                firsts = [0] + list(accumulate((len(token['form']) for token in sentence)))[:-1]
                length = len(chars)
                lasts = [(p-1) % length for p in firsts]

                sentences.append({
                    'chars': chars,
                    'words': words,
                    'firsts': firsts,
                    'lasts': lasts,
                    'tags': tags
                })

            with open(f'Datasets/{dataset}/{language}.json', 'w') as file:
                json.dump(sentences, file, indent=4)
            print(' done.')


if __name__ == "__main__":
    # generate_enumerations()
    convert_data()
