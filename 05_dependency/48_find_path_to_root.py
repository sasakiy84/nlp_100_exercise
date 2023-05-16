from chunk_sentence import Chunk
from typing import Tuple

from morph import Morph
from chunk_sentence import Sentece
import load_data

root = load_data.load()

for sentence_node in root:

    if len(sentence_node) == 0:
        continue
    sentence = Sentece(sentence_node)

    for chunk in sentence.chunks:
        is_noun_chunk = False
        for morph in chunk.morphs:
            if morph.pos == "名詞":
                is_noun_chunk = True
                break

        if is_noun_chunk == False:
            continue

        path_to_root = sentence.gather_chunks_to_root(chunk.id)
        print(" -> ".join(map(lambda v: v.to_text(), path_to_root)))
