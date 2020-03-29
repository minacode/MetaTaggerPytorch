from build_dicts import ID, FORM, tag_name_to_column
from Corpora.ud_test_v2_0_conll2017.evaluation_script.conll17_ud_eval import evaluate, load_conllu_file
import torch


def evaluate_model(model, tag_name, path, labeled_data):
    model.eval()

    work_data = load_conllu_file(path)
    words_count = len(work_data.tokens)

    tag_column_id = tag_name_to_column(tag_name=tag_name)

    # print(f'words: {words_count}')

    # TODO make a function ouf of this that is also called in build_dicts
    start_id = 0
    while start_id < words_count:
        end_id = start_id + 1
        while end_id < words_count and work_data.words[end_id].columns[ID] != '1':
            end_id += 1

        # print(start_id, end_id)

        char_ids = []
        word_ids = []
        first_ids = []
        last_ids = []
        char_pos = 0

        # end_word_id stops on first token of next sentence
        for token in work_data.words[start_id: end_id]:
            token_chars = [
                labeled_data.lexicon.get_char(char)
                for char
                in work_data.characters[token.span.start: token.span.end]
            ]
            char_ids.extend(token_chars)
            char_ids.append(labeled_data.lexicon.get_char(' '))
            first_ids.append(char_pos)
            char_pos += len(token_chars)
            last_ids.append(char_pos - 1)
            # add one for space between words
            char_pos += 1
            word_ids.append(
                labeled_data.lexicon.get_word(token.columns[FORM])
            )

        probabilities = model([
            torch.tensor(char_ids, dtype=torch.long, device=model.device),
            torch.tensor(word_ids, dtype=torch.long, device=model.device),
            torch.tensor(last_ids, dtype=torch.long, device=model.device),
            torch.tensor(first_ids, dtype=torch.long, device=model.device),
        ])

        # TODO zip Tensor? make this better
        for tag_id, token in zip(
                probabilities.argmax(dim=1),
                work_data.words[start_id: end_id]
        ):
            token.columns[tag_column_id] = labeled_data.tags[tag_name].get_value(tag_id)
            # print(token.columns)

        start_id = end_id

    gold_data = load_conllu_file(path)
    scores = evaluate(gold_data, work_data)

    # TODO maybe unpack this foreign scores
    return scores
