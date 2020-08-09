import sys
sys.path.append('..')
import numpy as np
from data import ptb
from util import *

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = preprocessing("you say goodbye and I say hello")
vocab_size = len(word_to_id)
co_matrix = create_co_matrix(corpus, vocab_size, window_size)
ppmi_mat = ppmi(co_matrix, svd=True, wordvec_size=2)
for query in ['you']:
    most_similar(query, word_to_id, id_to_word, ppmi_mat, top=5)