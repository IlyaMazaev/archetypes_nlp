from natasha import (
    Segmenter, MorphVocab,
    NewsNERTagger,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import re
import glob
import tqdm


def prepare_russian_text(input_file, output_file):
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)
    morph_vocab = MorphVocab()

    label_dict = {'NUM': 'ordinal1', 'PRON': 'pron1', 'PER': 'person1'}
    next_label_num = 5

    prepared_text = ''

    with open(input_file, encoding='utf-8') as fin:
        raw_text_lines = fin.readlines()

    for raw_text in raw_text_lines:
        raw_text = re.sub(r'\d+', '0', raw_text)

        doc = Doc(raw_text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        for span in reversed(doc.ner.spans):
            if span.type not in label_dict:
                label_dict[span.type] = str(next_label_num)
                next_label_num += 1
            raw_text = "".join((raw_text[:span.start], label_dict[span.type], raw_text[span.stop:]))

        doc = Doc(raw_text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        prev_num = False
        for token in doc.tokens:

            if token.pos == 'NUM' and not token.text.isdigit():
                if not prev_num:
                    prepared_text += '0'
                    prepared_text += ' '
                    prev_num = True
                continue

            prev_num = False

            if token.pos in label_dict:
                prepared_text += label_dict[token.pos]
                prepared_text += ' '

            elif token.pos != 'PUNCT':
                try:
                    token.lemmatize(morph_vocab)
                    prepared_text += token.lemma.lower()
                    prepared_text += ' '
                except Exception as ex:
                    prepared_text += token.text.lower()
                    prepared_text += ' '

        prepared_text += '\n'

    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write(prepared_text)


import time

start_time = time.time()

prepare_russian_text('combined_text_lower.txt', 'combined_text_lower_normalized.txt')

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
