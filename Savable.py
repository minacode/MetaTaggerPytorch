import json
from abc import ABCMeta, abstractmethod


class Savable(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dict):
        pass

    def save(self, path):
        with open(path, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as file:
            dict = json.load(file)
        return cls.from_dict(dict)
