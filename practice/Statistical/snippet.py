import sys
sys.path.append('..')
from common.util import *
import torch
text = 'you say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
context, target = create_context_target(corpus, window_size=1)
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
context = convert_one_hot(context, vocab_size)
print(target)
print(context)