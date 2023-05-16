
from chunk_sentence import Chunk
from typing import Tuple

from morph import Morph
from chunk_sentence import Sentece
import load_data

root = load_data.load()

print("print dependency as normal text")
for sentence_node in root:

    if len(sentence_node) == 0:
        continue
    sentence = Sentece(sentence_node)

    # find verb chunk
    verb_chunks: list[Tuple[Chunk, Morph]] = []
    for chunk in sentence.chunks:

        for morph in chunk.morphs:
            if morph.pos == "動詞":
                verb_chunks.append((chunk, morph))
                break

    if len(verb_chunks) == 0:
        continue

    # find chunk depending on verb chunk and extract cases
    for verb_chunk, verb_morph in verb_chunks:
        src_chunks = sentence.find_chunks_by_dst_id(verb_chunk.id)

        # change below from question 45
        cases: list[Tuple[str, Chunk]] = []
        for chunk in src_chunks:
            for morph in chunk.morphs:
                if morph.pos == "助詞":
                    cases.append((morph.surface, chunk))

        cases.sort(key=lambda v: v[0])
        tab_separeted_cases = "\t".join(map(lambda v: v[0], cases))
        tab_separeted_chunks = "\t".join(map(lambda v: v[1].to_text(), cases))

        print(f"{verb_morph.base}\t{tab_separeted_cases}\t{tab_separeted_chunks}")
