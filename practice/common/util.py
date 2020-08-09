import re
import torch
import matplotlib.pyplot as plt
def corpus2dict(corpus_raw):
    word_to_id = {}
    id_to_word = {}
    for word in corpus_raw:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    return word_to_id, id_to_word

def corpus2Id(corpus_raw, w2i):
    return torch.tensor([w2i[w] for w in corpus_raw], dtype=torch.int32)

def preprocess(corpus_raw, subset=1.0):
    corpus_raw = re.sub("[^\w]", " ",  corpus_raw.lower()).split()
    corpus_raw = corpus_raw[0:int(len(corpus_raw)*subset)]
    word_to_id, id_to_word = corpus2dict(corpus_raw)
    corpus_id = corpus2Id(corpus_raw, word_to_id)
    return corpus_id, word_to_id, id_to_word

def create_co_matrix(corpus_id, vocab_size, window_size=1):
    corpus_id_len = len(corpus_id)
    matrix = torch.zeros((vocab_size, vocab_size))
    words = corpus_id.unique()
    for word in words:
        index = (corpus_id == word).nonzero()
        for i in index:
            for j in range(i-window_size, i+window_size+1):
                if j < 0 or j == i or j >= corpus_id_len:
                    continue

                matrix[word,corpus_id[j]] += 1
    return matrix

def cos_similarity(x, y):
    eps = 1e-08
    norm_x = x / torch.sqrt(torch.sum(x**2) + eps)
    norm_y = y / torch.sqrt(torch.sum(y**2) + eps)
    return torch.dot(norm_x, norm_y)

def most_similar(word, word_to_id, id_to_word, word_matrix, top=5):
    assert(word in word_to_id)
    size = len(word_to_id)
    similarity = torch.zeros(size)
    for i in range(size):
        if i == word_to_id[word]:
            continue
        similarity[i] = cos_similarity(word_matrix[word_to_id[word]], word_matrix[i])
    for i in similarity.argsort(descending=True):
        print("%s : %.4f" % (id_to_word[i.item()], similarity[i]))

def ppmi(matrix, eps=1e-08, svd=False, wordvec_size=2):
    ppmi_m = torch.zeros_like(matrix)
    N = torch.sum(matrix)
    A = torch.sum(matrix, axis=0)
    vocab_size = matrix.shape[0]
    for i in range(vocab_size):
        for j in range(vocab_size):
            ppmi_m[i][j] = max(0, torch.log2(matrix[i][j] * N / (A[i] * A[j]) + eps))
    if svd is True:
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=wordvec_size, n_iter=5)
            svd.fit(ppmi_m)
            new = torch.from_numpy(svd.transform(ppmi_m))
        except ImportError:
            new, _, _ = torch.svd(ppmi_m)
        return new[:, :wordvec_size]
    else:
        return ppmi_m
    
def similarity_plt(word_to_id, matrix):
    for word, word_id in word_to_id.items():
        plt.annotate(word, (matrix[word_id, 0], matrix[word_id, 1]))
    plt.scatter(matrix[:, 0], matrix[:, 1], alpha=0.5)
    plt.show()

def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = torch.zeros((N, vocab_size), dtype=torch.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = torch.zeros((N, C, vocab_size), dtype=torch.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def create_context_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    context = torch.zeros(2 * window_size).int()
    for i in range(window_size, len(corpus)-window_size):
        sub = corpus[i-window_size:i+window_size+1].clone()
        sub = torch.cat([sub[0:window_size], sub[window_size+1:sub.shape[0]]])
        context = torch.cat((context, sub))
    context = context.reshape((-1, window_size*2))
    return context[1:], target

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate