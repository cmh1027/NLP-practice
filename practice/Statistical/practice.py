
import sys
sys.path.append('..')
import torch
from data import ptb
from common.util import most_similar, create_co_matrix, ppmi

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus = torch.from_numpy(corpus)
vocab_size = len(word_to_id)
co_matrix = create_co_matrix(corpus, vocab_size, window_size)
ppmi_mat = ppmi(co_matrix, svd=True, wordvec_size=wordvec_size)
for query in ['you', 'year', 'car', 'toyota']:
    most_similar(query, word_to_id, id_to_word, ppmi_mat, top=5)