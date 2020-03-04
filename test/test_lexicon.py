import unittest
from Lexicon import Singleton, Enumerator, Lexicon, LabeledData, get_labeled_data_path


class SingletonTest(unittest.TestCase):
    def test_unique(self):
        class A(metaclass=Singleton):
            pass

        a1 = A()
        a2 = A()
        self.assertIs(a1, a2)


class EnumeratorTest(unittest.TestCase):
    def test_basic(self):
        enumerator = Enumerator()
        enumerator.add('a')
        enumerator.add('b')
        enumerator.add('c')
        self.assertEqual(
            enumerator['a'],
            0
        )
        self.assertEqual(
            enumerator['b'],
            1
        )
        self.assertEqual(
            enumerator['c'],
            2
        )

    def test_double_add(self):
        e = Enumerator()
        e.add('a')
        a1 = e['a']
        e.add('a')
        a2 = e['a']
        self.assertEqual(a1, a2)
        self.assertEqual(
            len(e),
            1
        )

    def test_double_add_unknown(self):
        e = Enumerator(has_unknown=True)
        e.add('a')
        a1 = e['a']
        e.add('a')
        a2 = e['a']
        self.assertEqual(a1, a2)
        self.assertEqual(
            len(e),
            2
        )

    def test_basic_with_unknown(self):
        enumerator = Enumerator(has_unknown=True)
        enumerator.add('a')
        enumerator.add('b')
        enumerator.add('c')
        self.assertEqual(
            enumerator['a'],
            1
        )
        self.assertEqual(
            enumerator['b'],
            2
        )
        self.assertEqual(
            enumerator['c'],
            3
        )

    def test_unknown(self):
        enumerator = Enumerator(has_unknown=True)
        self.assertEqual(
            enumerator['a'],
            0
        )
        enumerator.add('a')
        self.assertEqual(
            enumerator['a'],
            1
        )

    def test_idempotence(self):
        e1 = Enumerator()
        for c in 'uiatreundiaternuiae':
            e1.add(c)
        export = e1.to_dict()
        e2 = Enumerator.from_dict(export)
        self.assertEqual(e1, e2)


class LexiconTest(unittest.TestCase):
    def test_idempotence(self):
        lexicon1 = Lexicon()
        for word in ['uiae', 'flhcws', 'envfchw', 'trnöaes']:
            lexicon1.add_word(word)
        export = lexicon1.to_dict()
        lexicon2 = Lexicon.from_dict(export)
        self.assertEqual(lexicon1, lexicon2)


class LabeledDataTest(unittest.TestCase):
    def test_idempotence(self):
        labeled_data1 = LabeledData.load(
            get_labeled_data_path(
                dataset='conll17',
                language='de'
            )
        )
        export = labeled_data1.to_dict()
        labeled_data2 = LabeledData.from_dict(export)
        self.assertEqual(labeled_data1, labeled_data2)


if __name__ == '__main__':
    unittest.main()
