
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

        valid_noun_chunks: list[Tuple[str, Chunk]] = []
        for chunk in src_chunks:
            mode = "SEARCHING_SAHEN_NOUN"
            previous_morph = None
            for morph in chunk.morphs:
                if morph.pos1 == "サ変名詞":
                    mode = "SEARCHING_WO"
                elif mode == "SEARCHING_WO" and morph.surface == "を":
                    verb_phrase = f"{previous_morph.surface}{morph.surface}{verb_morph.base}"
                    sahen_noun_chunk = chunk
                    valid_noun_chunks.append((verb_phrase, sahen_noun_chunk))
                elif mode == "SEARCHING_WO":
                    mode = "SEARCHING_SAHEN_NOUN"

                previous_morph = morph

        for verb_phrase, sahen_noun_chunk in valid_noun_chunks:
            src_chunks_of_sahen = sentence.find_chunks_by_dst_id(
                sahen_noun_chunk.id)

            cases: list[Tuple[str, Chunk]] = []
            for chunk in [*src_chunks, *src_chunks_of_sahen]:
                if chunk.id == sahen_noun_chunk.id:
                    continue
                for morph in chunk.morphs:
                    if morph.pos == "助詞":
                        cases.append((morph.surface, chunk))

            if len(cases) == 0:
                continue

            cases.sort(key=lambda v: v[0])
            space_separated_cases = " ".join(map(lambda v: v[0], cases))
            space_separated_chunks = " ".join(
                map(lambda v: v[1].to_text(), cases))

            print(
                f"{verb_phrase} {space_separated_cases} {space_separated_chunks}")

# check with unix command
# python 47_mining.py | grep "学習を行う"
