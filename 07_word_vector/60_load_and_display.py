from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin.gz', binary=True)

print('Tuple Length: ' + str(len(model['United_States'])))
print(model['United_States'])
