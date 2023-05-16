"""
Design a class Word that represents a word. 
This class has three member variables, text (word surface), lemma (lemma), and pos (part-of-speech). 
Represent a sentence as an array of instances of Word class. 
Implement a program to load the parse result, and store the text as an array of sentences. 
Show the object of the first sentence of the body of the article.
https://nlp100.github.io/en/ch05.html#40-read-the-parse-result-words
"""

from utils import parse_token_to_morph
from chunk_sentence import Morph
import load_data

root = load_data.load()
result: list[list[Morph]] = []

for sentence in root:
    sentence_buffer = []
    for chunk in sentence:
        for token in chunk:
            sentence_buffer.append(parse_token_to_morph(token))
    result.append(sentence_buffer)


first_sentence = result[2]


for morph in first_sentence:
    print(morph.to_str())
