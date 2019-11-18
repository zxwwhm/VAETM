import gensim
import os
import pickle
import numpy as np

import file_handling as fh

# WORD2VEC_FILE = "/home/lcw2/share/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
#W2V_dim = 300
WORD2VEC_FILE = "/home/lcw2/share/embeddings/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin"
W2V_dim = 200
KB2VEC_FILE = "/home/lcw2/github/my_vaetm/data/kb2vec/WikiData.KB.100d.zh.v2.pickle"
K2V_DIM = 100

INPUT_DIR = '../scholar/data/standard/processed'
OUTPUT_TRAIN_FILE = 'dataset/standard.Tencent.WikiData.300d.train.pickle'
OUTPUT_VOC_EMB = 'dataset/standard.Tencent.WikiData.300d.voc_emb.pickle'
OUTPUT_VOC_FILE = 'dataset/standard.5000.vocabulary.pickle'

def main():
    train_X, vocab, col_sel = load_data(INPUT_DIR)
    n_train, vocab_size = train_X.shape
    print('Shape of train_X_raw:', train_X.shape)

    with open(OUTPUT_VOC_FILE, 'wb') as f:
        pickle.dump(vocab, f)
    print('Vocabulary saved to {}!'.format(OUTPUT_VOC_FILE))
    
    rng = np.random.RandomState(np.random.randint(0, 100000))
    w2v_embeddings = load_word2vec(vocab, vocab_size, rng)
    k2v_embeddings = load_knowledge2vec(vocab, vocab_size, rng)

    X_mult_w2v = np.matmul(train_X, w2v_embeddings)
    X_mult_k2v = np.matmul(train_X, k2v_embeddings)

    print('Shape of X_mult_w2v:', X_mult_w2v.shape)
    print('Shape of X_mult_k2v:', X_mult_k2v.shape)

    trainset = np.concatenate((X_mult_w2v, X_mult_k2v), axis=1)
    trainset = trainset/n_train
    print('Shape of trainset:', trainset.shape)
    print('trainset[0]:\n', trainset[0])

    with open(OUTPUT_TRAIN_FILE, 'wb') as f:
        pickle.dump(trainset, f)
    print('Matrix of dataset saved to {}!'.format(OUTPUT_TRAIN_FILE))

    embeddings = np.concatenate((w2v_embeddings, k2v_embeddings), axis=1)
    print('Shape of voc_embeddings:', embeddings.shape)
    with open(OUTPUT_VOC_EMB, 'wb') as f:
        pickle.dump(embeddings, f)
    print('Matrix of voc_embeddings saved to {}!'.format(OUTPUT_VOC_EMB))



def load_data(input_dir:str, input_prefix='train', vocab_size=None, vocab=None, col_sel=None):
    print("Loading data...")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    n_items, temp_size = temp.shape
    print("  Loaded %d documents with %d features" % (n_items, temp_size))

    if vocab is None:
        col_sel = None
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
        # filter vocabulary by word frequency
        if vocab_size is not None:
            print("  Filtering vocabulary to the most common %d terms" % int(vocab_size))
            col_sums = np.array(temp.sum(axis=0)).reshape((len(vocab), ))
            order = list(np.argsort(col_sums))
            order.reverse()
            col_sel = np.array(np.zeros(len(vocab)), dtype=bool)
            for i in range(int(vocab_size)):
                col_sel[order[i]] = True
            temp = temp[:, col_sel]
            vocab = [word for i, word in enumerate(vocab) if col_sel[i]]
    elif col_sel is not None:
        print("  Using given vocabulary")
        temp = temp[:, col_sel]

    X = np.array(temp, dtype='float32')
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("  Loaded %d documents with %d features" % (n_items, vocab_size))

    # filter out empty documents
    non_empty_sel = X.sum(axis=1) > 0
    print("  Found %d non-empty documents" % np.sum(non_empty_sel))
    X = X[non_empty_sel, :]

    counts_sum = X.sum(axis=0)
    order = list(np.argsort(counts_sum).tolist())
    order.reverse()
    print("  Most common words: ", ' '.join([vocab[i] for i in order[:10]]))

    return X, vocab, col_sel


def load_word2vec(vocab, vocab_size, rng):
    vocab_dict = dict(zip(vocab, range(vocab_size)))
    embeddings = np.array(rng.rand(vocab_size, W2V_dim) * 0.25 - 0.5, dtype=np.float32)
    count = 0
    print("Loading word vectors...")
    # pretrained = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True)
    pretrained = gensim.models.KeyedVectors.load(WORD2VEC_FILE)

    for word, index in vocab_dict.items():
        if word in pretrained:
            count += 1
            embeddings[index, :] = pretrained[word]
    print("  Found word embeddings for %d words" % count)
    print('  Shape of word embeddings:', embeddings.shape)
    return embeddings


def load_knowledge2vec(vocab, vocab_size, rng):
    vocab_size = len(vocab)
    vocab_dict = dict(zip(vocab, range(vocab_size)))
    embeddings = np.array(rng.rand(vocab_size, K2V_DIM) * 0.25 - 0.5, dtype=np.float32)
    count = 0
    print("Loading knowledge vectors...")
    pretrained = None
    with open(KB2VEC_FILE, 'rb') as f:
        pretrained = pickle.load(f)
    print('  # of entities:', len(pretrained))

    for word, index in vocab_dict.items():
        if word in pretrained:
            count += 1
            embeddings[index, :] = pretrained[word]
    print("  Found entity embeddings for %d words" % count)
    print('  Shape of knowledge embeddings:', embeddings.shape)
    return embeddings


if __name__ == '__main__':
    main()