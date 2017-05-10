import numpy as np

def load(path, vocab):
    emb_dict = None
    with open(path) as emb_f:
        for line in emb_f:
            parts = line.strip().split()
            word = parts[0]
            if word not in vocab.contents:
                continue
            vec = [float(f) for f in parts[1:]]
            if emb_dict is None:
                emb_dict = np.zeros((len(vocab), len(vec)))
            emb_dict[vocab[word], :] = vec
    return emb_dict
