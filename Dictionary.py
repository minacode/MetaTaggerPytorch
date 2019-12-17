import json
from conllu import parse


class Enumerator:
    def __init__(self):
        self._elements = {}
        self._n = 0

    # data contains [(value, index)]
    # TODO maybe check if input is well-formed
    @classmethod
    def from_dict(cls, data):
        enumerator = cls()
        enumerator._n = len(data)
        enumerator._elements = dict(zip(data, range(len(data))))
        return enumerator

    def to_dict(self):
        elements = list(self._elements.items())
        elements.sort(key=lambda x: x[1])
        return [x[0] for x in elements]

    def add(self, x):
        if x not in self._elements:
            self._elements[x] = self._n
            self._n += 1

    def get_value(self, index):
        for key in self._elements:
            if self._elements[key] == index:
                return key
        raise ValueError(f'{index} is no valid index')

    def get(self, x):
        if x in self._elements:
            return self._elements[x]
        raise KeyError(f'{x} is no valid element')

    def __getitem__(self, x):
        return self.get(x)

    def __len__(self):
        return self._n


class UnknownEnumerator(Enumerator):
    def __init__(self):
        Enumerator.__init__(self)
        self._elements = {self.unknown(): 0}
        self._n = 1

    def get(self, x):
        return self._elements[x] if x in self._elements else self._elements[self.unknown()]

    # represent UNKNOWN value, returns None, maybe do this explicit
    @staticmethod
    def unknown():
        pass


class Dictionary:
    def __init__(self):
        self._words = UnknownEnumerator()
        self._chars = UnknownEnumerator()

    @staticmethod
    def from_dict(data):
        dictionary = Dictionary()
        dictionary._chars = Enumerator.from_dict(data['chars'])
        dictionary._words = Enumerator.from_dict(data['words'])
        return dictionary

    def to_dict(self):
        return {
            'chars': self._chars.to_dict(),
            'words': self._words.to_dict()
        }

    def n_words(self):
        return len(self._words)

    def n_chars(self):
        return len(self._chars)

    def add_word(self, word):
        self._words.add(word)
        for char in word:
            self._chars.add(char)

    def get_word(self, word):
        return self._words[word]

    def get_char(self, char):
        return self._chars[char]

    def get_chars(self, word):
        return [
            self._chars[char]
            for char in word
        ]


# extend this with morphological tags
class LabeledData:
    def __init__(self, dataset, language, dictionary=None, tags=None):
        self.dataset = dataset
        self.language = language
        self.dictionary = dictionary if isinstance(dictionary, Dictionary) else Dictionary()
        self.tags = tags if isinstance(tags, Enumerator) else Enumerator()

    @classmethod
    def from_language(cls, language, dataset='conll17'):
        try:
            data = cls.load(cls.language_path(dataset, language))
            print(f'Language file for {dataset}: {language} loaded.')
            return data
        except FileNotFoundError:
            with open('data.json', 'r') as f:
                config = json.load(f)
            c = config[dataset]
            path = f'{c["base_path"]}/{c["train"]}/{c["file_name"].replace("<lang>", language)}'
            data = cls(
                dataset=dataset,
                language=language
            )
            data.update_from_file(path)
            print(f'Language file for {dataset}: {language} created.')
            return data

    # make a loadfile out of this
    def update_from_file(self, path):
        with open(path, 'r') as file:
            lines = file.read()
        sentences = parse(lines)
        for sentence in sentences:
            for token in sentence:
                self.dictionary.add_word(token['form'])
                self.tags.add(token['upostag'])

    @classmethod
    def from_dict(cls, data):
        return cls(
            dataset=data['dataset'],
            language=data['language'],
            dictionary=Dictionary.from_dict(data['dictionary']),
            tags=Enumerator.from_dict(data['tags'])
        )

    def to_dict(self):
        return {
            'dataset': self.dataset,
            'language': self.language,
            'dictionary': self.dictionary.to_dict(),
            'tags': self.tags.to_dict()
        }

    @staticmethod
    def directory_path(dataset):
        return f'Dictionaries/{dataset}'

    @staticmethod
    def language_path(dataset, language):
        return f'{LabeledData.directory_path(dataset)}/{language}.json'

    def save(self):
        directory_path = self.directory_path(self.dataset)
        with open(f'{directory_path}/{self.language}.json', 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as file:
            dict = json.load(file)
        return cls.from_dict(dict)
