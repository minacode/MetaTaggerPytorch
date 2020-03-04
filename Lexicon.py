from typing import List, Optional, Dict, TypeVar, Union, Any, Literal
import json
from conllu import parse
from Savable import Savable


T = TypeVar('T')


class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


class Unknown(metaclass=Singleton):
    pass


def unknown() -> Unknown:
    return Unknown()


# must have the same values
TAG_NAMES = ['POS', 'XPOS', 'FEATURE']


def tag_name_to_column(tag_name):
    return {
        'POS': 3,
        'XPOS': 4,
        'FEATURE': 5
    }[tag_name]


class Sentence(dict):
    char_ids: List[int]
    word_ids: List[int]
    first_ids: List[int]
    last_ids: List[int]
    tag_ids: Dict[str, List[int]]


class EnumeratorExport(dict):
    has_unknown: bool
    elements: List[str]


class LexiconExport(dict):
    chars: EnumeratorExport
    words: EnumeratorExport


class LabeledDataExport(dict):
    dataset: str
    language: str
    lexicon: LexiconExport
    tags: Dict[str, EnumeratorExport]


class Enumerator:
    def __init__(self, has_unknown: bool = False):
        self.has_unknown: bool = has_unknown
        self._elements: Dict[Union[Unknown, str], int] = \
            {unknown(): 0} \
            if has_unknown \
            else {}
        self._n: int = 1 if has_unknown else 0

    def add(self, x: str) -> None:
        if x not in self._elements:
            self._elements[x] = self._n
            self._n += 1

    def get(self, x: str) -> int:
        if x in self._elements:
            return self._elements[x]
        elif self.has_unknown:
            return self._elements[unknown()]
        raise KeyError(f'{repr(x)} is no valid element')

    def get_value(self, index: int) -> Union[Unknown, str]:
        for key in self._elements:
            if self._elements[key] == index:
                return key
        raise ValueError(f'{index} is no valid index')

    def __getitem__(self, x: str) -> int:
        return self.get(x)

    def __len__(self) -> int:
        return self._n

    def __eq__(self, other):
        if not isinstance(other, Enumerator):
            return False
        return self._elements == other._elements

    def __repr__(self):
        return f'\n{self.has_unknown}\n{self._elements}'

    # data contains [(value, index)]
    # TODO maybe check if input is well-formed
    @classmethod
    def from_dict(cls, data: EnumeratorExport):
        enumerator = cls(has_unknown=data['has_unknown'])
        elements = data['elements']
        for element in elements:
            enumerator.add(element)
        return enumerator

    def to_dict(self) -> EnumeratorExport:
        elements = list(self._elements.items())
        elements.sort(key=lambda x: x[1])
        return EnumeratorExport(
            has_unknown=self.has_unknown,
            elements=[
                x[0]
                for x in elements
                if not isinstance(x[0], Unknown)
            ]
        )


class Lexicon:
    def __init__(self) -> None:
        self._chars: Enumerator = Enumerator(has_unknown=True)
        self._words: Enumerator = Enumerator(has_unknown=True)

    @classmethod
    def from_dict(cls, data: LexiconExport):
        lexicon = cls()
        lexicon._chars = Enumerator.from_dict(data['chars'])
        lexicon._words = Enumerator.from_dict(data['words'])
        return lexicon

    def to_dict(self) -> LexiconExport:
        return LexiconExport(
            chars=self._chars.to_dict(),
            words=self._words.to_dict()
        )

    def n_words(self) -> int:
        return len(self._words)

    def n_chars(self) -> int:
        return len(self._chars)

    def add_word(self, word: str) -> None:
        self._words.add(word)
        for char in word:
            self._chars.add(char)

    def get_word(self, word: str) -> int:
        return self._words[word]

    def get_word_by_id(self, i: int) -> Union[Unknown, str]:
        return self._words.get_value(i)

    def add_char(self, char: str) -> None:
        self._chars.add(char)

    def get_char(self, char: str) -> int:
        return self._chars.get(char)

    def get_char_by_id(self, i: int) -> Union[Unknown, str]:
        return self._chars.get_value(i)

    def get_chars(self, word: str) -> List[int]:
        return [
            self._chars[char]
            for char in word
        ]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Lexicon):
            return False
        return all([
            self._chars == other._chars,
            self._words == other._words
        ])


# extend this with morphological tags
class LabeledData(Savable):
    def __init__(
            self,
            dataset: str,
            language: str,
            labels: List[str],
            lexicon: Optional[Lexicon] = None,
            tags: Optional[Dict[str, Enumerator]] = None
    ) -> None:
        self.dataset: str = dataset
        self.language: str = language
        self.labels: List[str] = labels
        self.lexicon: Lexicon = lexicon if lexicon is not None else Lexicon()
        self.tags: Dict[str, Enumerator] = tags if tags is not None else {
            tag_name: Enumerator()
            for tag_name in TAG_NAMES
        }

    def __eq__(self, other: Any) -> bool:
        return all([
            self.dataset == other.dataset,
            self.language == other.language,
            self.labels == other.labels,
            self.lexicon == other.lexicon,
            self.tags == other.tags
        ])

    @classmethod
    def create_from_language(cls, language: str, dataset: str = 'conll17'):
        with open('data.json', 'r') as f:
            config = json.load(f)
        c = config[dataset]
        path = f'{c["base_path"]}/{c["train"]}/{c["file_name"].replace("<lang>", language)}'
        data = cls(
            dataset=dataset,
            language=language,
            labels=TAG_NAMES
        )
        data.update_from_file(path)
        print(f'Language file for {dataset}: {language} created.')
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            dataset=data['dataset'],
            language=data['language'],
            labels=TAG_NAMES,
            lexicon=Lexicon.from_dict(data['lexicon']),
            tags={
                label: Enumerator.from_dict(data['tags'][label])
                for label in TAG_NAMES
            }
        )

    def to_dict(self) -> LabeledDataExport:
        return LabeledDataExport(
            dataset=self.dataset,
            language=self.language,
            lexicon=self.lexicon.to_dict(),
            tags={
                label: self.tags[label].to_dict()
                for label
                in self.tags
            }
        )

    # make a loadfile out of this
    def update_from_file(self, path: str) -> None:
        with open(path, 'r') as file:
            lines = file.read()
        # TODO kill it with fire!!!1k!!1¹!
        sentences = parse(lines)
        for sentence in sentences:
            for token in sentence:
                self.lexicon.add_word(token['form'])
                # TODO add features. parser does not find them?
                for label in self.labels:
                    if label not in self.tags:
                        self.tags[label] = Enumerator()
                    # TODO maybe handle that some values are labeled with None
                    # the UnknownEnumerator checks whether a collision with logical "unknown" happens via .add
                    self.tags[label].add(token[label])


def get_labeled_data_path(dataset: str, language: str) -> str:
    return f'Dictionaries/{dataset}/{language}.json'
