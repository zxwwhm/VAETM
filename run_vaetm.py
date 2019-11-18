import collections
import os
import sys
from optparse import OptionParser

import gensim
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

import file_handling as fh

from vaetm import VaeTm


def main():
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-k', dest='n_topics', default=50,
                      help='Size of latent representation (~num topics): default=%default')
    parser.add_option('-r', action="store_true", dest="regularize", default=False,
                      help='Apply adaptive regularization for sparsity in topics: default=%default')
    parser.add_option('-o', dest='output_dir', default='output',
                      help='Output directory: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Filter the vocabulary keeping the most common n words: default=%default')
    parser.add_option('--no-bg', action="store_true", dest="no_bg", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--no-bn-anneal', action="store_true", dest="no_bn_anneal", default=False,
                      help='Do not anneal away from batchnorm: default=%default')
    parser.add_option('--opt', dest='optimizer', default='adam',
                      help='Optimization algorithm to use [adam|adagrad|sgd]: default=%default')
    parser.add_option('--dev-folds', dest='dev_folds', default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', dest='dev_fold', default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default=None,
                      help='Prefix of test set: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]

    dev_folds = int(options.dev_folds)
    dev_fold = int(options.dev_fold)

    alpha = 1.0
    n_topics = int(options.n_topics)
    batch_size = 200
    # learning_rate = 0.002
    learning_rate = 0.001
    adam_beta1 = 0.99
    n_epochs = 200
    encoder_layers = 2 #Number of encoder layers [0|1|2]
    encoder_shortcuts = False
    classifier_layers = 1 #[0|1|2]
    auto_regularize = options.regularize
    output_dir = options.output_dir
    word2vec_file = "../embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
    # word2vec_file = "C:\\\\Soft\\share\\GoogleNews-vectors-negative300.bin"
    embedding_dim = 300
    vocab_size = options.vocab_size
    update_background = False
    no_bg = options.no_bg
    bn_anneal = not options.no_bn_anneal
    optimizer = options.optimizer
    seed = 1
    threads = 4
    if seed is not None:
        seed = int(seed)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
    # kb embedding file
    kb2vec_file = "data/kb2vec/WikiData.KB.100d.pickle"
    kb_dim = 100
    test_prefix = 'test'

    # load the training data
    train_prefix = 'train'
    train_X, vocab, col_sel ,num= load_data(input_dir, train_prefix, vocab_size=vocab_size)
    n_train, dv = train_X.shape

    if test_prefix=='test':
        test_X, _, _ ,_= load_data(input_dir, test_prefix, vocab=vocab)
        n_test, _ = test_X.shape


    # split training data into train and dev
    if dev_folds > 0:
        n_dev = int(n_train / dev_folds)
        indices = np.array(range(n_train), dtype=int)
        rng.shuffle(indices)
        if dev_fold < dev_folds - 1:
            dev_indices = indices[n_dev * dev_fold: n_dev * (dev_fold +1)]
        else:
            dev_indices = indices[n_dev * dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        n_train = len(train_indices)
    else:
        dev_X = None

    # initialize the background using the overall frequency of terms
    init_bg = get_init_bg(train_X)
    init_beta = None
    update_beta = True
    # if no_bg:
    #     if n_topics == 1:
    #         init_beta = init_bg.copy()
    #         init_beta = init_beta.reshape([1, len(vocab)])
    #         update_beta = False
    #     init_bg = np.zeros_like(init_bg)

    # create the network configuration
    network_architecture = make_network(dv, encoder_layers, embedding_dim,
                                        n_topics, encoder_shortcuts,
                                        classifier_layers)

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # # load pretrained word vectors
    if word2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        embeddings = np.array(rng.rand(vocab_size, embedding_dim) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        if word2vec_file[-3:] == 'bin':
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[index, :] = pretrained[word]

        print("Found word embeddings for %d words" % count)
        print('shape of word embeddings:', embeddings.shape)
    else:
        print("No embeddings for words!")
        exit()

    # # load pretrained entity vectors
    if kb2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        entity_embeddings = np.array(rng.rand(vocab_size, kb_dim) * 0.25 - 0.5, dtype=np.float32)
        count = 0

        print("Loading emtity vectors...")
        pretrained = None
        with open(kb2vec_file, 'rb') as f:
            pretrained = pickle.load(f)
        print('# of entities:', len(pretrained))
        vocab_counter = collections.Counter()
        vocab_counter.update(s for s in num if s in pretrained)
        print(vocab_counter.most_common(30))
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                entity_embeddings[index, :] = pretrained[word]

        print("Found entity embeddings for %d words" % count)
        print('shape of entity embeddings:', entity_embeddings.shape)
    else:
        print("No embeddings for knowledge entities!")
        exit()

    tf.reset_default_graph()

    # create the model
    model = VaeTm(network_architecture, alpha=alpha,\
        learning_rate=learning_rate, \
        batch_size=batch_size, init_embeddings=embeddings,\
        entity_embeddings=entity_embeddings,\
        init_bg=init_bg,\
        update_background=update_background, init_beta=init_beta,\
        update_beta=update_beta, threads=threads,\
        regularize=auto_regularize, optimizer=optimizer,\
        adam_beta1=adam_beta1, seed=seed)

    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, train_X, vocab, regularize=auto_regularize, training_epochs=n_epochs, batch_size=batch_size, rng=rng, bn_anneal=bn_anneal, X_dev=dev_X)

    # create output directory
    fh.makedirs(output_dir)

    # print background
    bg = model.get_bg()
    if not no_bg:
        print_top_bg(bg, vocab)

    # print topics
    emb = model.get_weights()
    print("Topics:")
    maw, sparsity, topics = print_top_words(emb, vocab)
    print("sparsity in topics = %0.4f" % sparsity)
    save_weights(output_dir, emb, bg, vocab, sparsity_threshold=1e-5)

    fh.write_list_to_text(['{:.4f}'.format(maw)], os.path.join(output_dir, 'maw.txt'))
    fh.write_list_to_text(['{:.4f}'.format(sparsity)], os.path.join(output_dir, 'sparsity.txt'))

    # print('Predicting training representations...')
    # reps, preds = model.predict(train_X, None)
    # # print('rep-0:', reps[0])
    # # print('rep-0:', reps[1])
    # fh.write_matrix_to_text(reps, os.path.join(output_dir, 'train_representation.txt'))

    # if test_X is not None:
        # print('Predicting testing representations...')
        # reps, preds = model.predict(test_X)
        # # print('rep-0:', reps[0])
        # # print('rep-0:', reps[1])
        # fh.write_matrix_to_text(reps, os.path.join(output_dir, 'test_representation.txt'))

    # Evaluate perplexity on dev and test dataa
    if dev_X is not None:
        perplexity = evaluate_perplexity(model, dev_X, eta_bn_prop=0.0)
        print("Dev perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.dev.txt'))

    if test_X is not None:
        perplexity = evaluate_perplexity(model, test_X, eta_bn_prop=0.0)
        print("Test perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.test.txt'))

    # save document representations
    theta = model.compute_theta(train_X, None)
    np.savez(os.path.join(output_dir, 'theta.train.npz'), theta=theta)
    compute_npmi_at_n(topics, vocab, train_X)


def load_data(input_dir:str, input_prefix:str, vocab_size=None, vocab=None, col_sel=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    n_items, temp_size = temp.shape
    print("Loaded %d documents with %d features" % (n_items, temp_size))

    if vocab is None:
        col_sel = None
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
        # filter vocabulary by word frequency
        if vocab_size is not None:
            print("Filtering vocabulary to the most common %d terms" % int(vocab_size))
            col_sums = np.array(temp.sum(axis=0)).reshape((len(vocab), ))
            order = list(np.argsort(col_sums))
            order.reverse()
            col_sel = np.array(np.zeros(len(vocab)), dtype=bool)
            for i in range(int(vocab_size)):
                col_sel[order[i]] = True
            temp = temp[:, col_sel]
            vocab = [word for i, word in enumerate(vocab) if col_sel[i]]

    elif col_sel is not None:
        print("Using given vocabulary")
        temp = temp[:, col_sel]

    X = np.array(temp, dtype='float32')
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    # filter out empty documents
    non_empty_sel = X.sum(axis=1) > 0
    print("Found %d non-empty documents" % np.sum(non_empty_sel))
    X = X[non_empty_sel, :]

    counts_sum = X.sum(axis=0)
    order = list(np.argsort(counts_sum).tolist())
    order.reverse()
    print("Most common words: ", ' '.join([vocab[i] for i in order[:10]]))
    num = list(vocab[i] for i in order[:200])
    return X, vocab, col_sel,num


def get_init_bg(data):
    """
    Compute the log background frequency of all words
    """
    sums = np.sum(data, axis=0)+1
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def create_minibatch(X, batch_size=200, rng=None):
    """
    Split data into minibatches
    """
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)
        yield X[ixs, :].astype('float32')


def make_network(dv, encoder_layers=2, embedding_dim=300, n_topics=50, encoder_shortcut=False, classifier_layers=1):
    """
    Combine the network configuration parameters into a dictionary
    """
    tf.reset_default_graph()
    network_architecture = \
        dict(encoder_layers=encoder_layers,
             encoder_shortcut=encoder_shortcut,
             embedding_dim=embedding_dim,
             n_topics=n_topics,label_emb_dim=-1,
             n_labels=0,
             dv=dv,
             classifier_layers=classifier_layers,
             )
    return network_architecture


#     model = train(model, network_architecture, train_X, regularize=auto_regularize, training_epochs=n_epochs, batch_size=batch_size, rng=rng, bn_anneal=bn_anneal, X_dev=dev_X)
def train(model, network_architecture, X, vocab, batch_size=200, training_epochs=100, display_step=5, min_weights_sq=1e-7, regularize=False, bn_anneal=True, init_eta_bn_prop=1.0, rng=None, X_dev=None):

    n_train, dv = X.shape
    mb_gen = create_minibatch(X, batch_size=batch_size, rng=rng)

    dv = network_architecture['dv']
    n_topics = network_architecture['n_topics']

    total_batch = int(n_train / batch_size)

    # create np arrays to store regularization strengths, which we'll update outside of the tensorflow model
    if regularize:
        l2_strengths = 0.5 * np.ones([n_topics, dv]) / float(n_train)
    else:
        l2_strengths = np.zeros([n_topics, dv])

    batches = 0

    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon
    # kld_weight = 1.0  # could use this to anneal KLD, but not currently doing so
    kld_weights = frange_cycle_linear(training_epochs, ratio=0.5)
    print('bn_anneal=', bn_anneal)

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        avg_cls_loss = 0.
        accuracy = 0.
        kld_weight = kld_weights[epoch]

        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs = next(mb_gen)
            # do one update, passing in the data, regularization strengths, and bn
            loss, cls_loss, pred = model.fit(batch_xs, None, l2_strengths=l2_strengths, eta_bn_prop=eta_bn_prop, kld_weight=kld_weight)
            # Compute average loss
            avg_loss += loss / n_train * batch_size
            avg_cls_loss += cls_loss / n_train * batch_size
            batches += 1
            if np.isnan(avg_loss):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

        # update weight prior variances using current weight values
        if regularize:
            weights = model.get_weights()
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_strengths = 0.5 / weights_sq / float(n_train)


        # Display logs per epoch step
        if epoch % display_step == 0 and epoch > 0:
            print("Epoch:", '%d' % epoch, "loss=", "{:.9f}".format(avg_loss))
            # emb = model.get_weights()
            # maw, sparsity, topics = print_top_words(emb, vocab)
            # compute_npmi_at_n(topics, vocab, X)

            if X_dev is not None:
                dev_perplexity = evaluate_perplexity(model, X_dev, eta_bn_prop=eta_bn_prop)
                print("  Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))



        # anneal eta_bn_prop from 1 to 0 over the course of training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(training_epochs*0.75)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

    return model


def predict_labels(model, X, C, eta_bn_prop=0.0):
    """
    Predict a label for each instance using the classifier part of the network
    """
    n_items, vocab_size = X.shape
    predictions = np.zeros(n_items, dtype=int)

    # predict items one by one
    for i in range(n_items):
        X_i = np.expand_dims(X[i, :], axis=0)
        # optionally provide covariates
        if C is not None:
            C_i = np.expand_dims(C[i, :], axis=0)
        else:
            C_i = None

        # predict probabilities
        z, y_recon = model.predict(X_i, C_i, eta_bn_prop=eta_bn_prop)

        # take the label with the maximum predicted probability
        pred = np.argmax(y_recon)
        predictions[i] = pred

    return predictions


def print_top_words(beta, feature_names, topic_names=None, n_top_words=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    topics = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        topic = []
        # get the top words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                topic.append(feature_names[order[j]])
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        output += ' / '
        topics.append(topic)

        # get the bottom words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += ': MAW=%0.4f' % maw + '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        # print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals), topics


def print_top_bg(bg, feature_names, n_top_words=10):
    """
    Print the most highly weighted words in the background log frequency
    """
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))


def evaluate_perplexity(model, X, eta_bn_prop=1.0, n_samples=0):
    """
    Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    """
    # count the number of words in each document
    doc_sums = np.array(X.sum(axis=1), dtype=float)
    X = X.astype('float32')
    # get the losses for all instances
    losses = model.get_losses(X, None, eta_bn_prop=eta_bn_prop, n_samples=n_samples)
    # compute perplexity for all documents in a single batch
    perplexity = np.exp(np.mean(losses / doc_sums))

    return perplexity


def save_weights(output_dir, beta, bg, feature_names, sparsity_threshold=1e-5):
    """
    Save model weights to npz files (also the top words in each topic
    """
    np.savez(os.path.join(output_dir, 'beta.npz'), beta=beta)
    if bg is not None:
        np.savez(os.path.join(output_dir, 'bg.npz'), bg=bg)
    fh.write_to_json(feature_names, os.path.join(output_dir, 'vocab.json'), sort_keys=False)

    topics_file = os.path.join(output_dir, 'topics.txt')
    lines = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words = [feature_names[j] for j in order[:40] if beta[i][j] > sparsity_threshold]
        output = ' '.join(pos_words)
        lines.append(output)

    fh.write_list_to_text(lines, topics_file)


def print_label_embeddings(model, class_names):
    """
    Display label embeddings
    """
    label_embeddings = model.get_label_embeddings()
    n_labels, _ = label_embeddings.shape
    dists = np.zeros([n_labels, n_labels])
    for i in range(n_labels):
        for j in range(n_labels):
            emb_i = label_embeddings[i, :]
            emb_j = label_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_labels):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = class_names[i] + ': '
        for j in range(4):
            output += class_names[order[j]] + ' '
        print(output)


def print_covariate_embeddings(model, covariate_names, output_dir):
    """
    Display covariate embeddings
    """
    covar_embeddings = model.get_covar_embeddings()
    n_covariates , emb_dim = covar_embeddings.shape
    dists = np.zeros([n_covariates, n_covariates])
    for i in range(n_covariates):
        for j in range(n_covariates):
            emb_i = covar_embeddings[i, :]
            emb_j = covar_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_covariates):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = covariate_names[i] + ': '
        for j in range(4):
            output += covariate_names[order[j]] + ' '
        print(output)
    if n_covariates < 30 and emb_dim < 10:
        print(covar_embeddings)
    np.savez(os.path.join(output_dir, 'covar_emb.npz'), W=covar_embeddings, names=covariate_names)


def predict_labels_and_evaluate(model, X, Y, C, output_dir=None, subset='train'):
    """
    Predict labels for all instances using the classifier network and evaluate the accuracy
    """
    n_items, vocab_size = X.shape
    predictions = predict_labels(model, X, C)
    accuracy = float(np.sum(predictions == np.argmax(Y, axis=1)) / float(n_items))

    print(subset, "accuracy on labels = %0.4f" % accuracy)
    # save the results to file
    if output_dir is not None:
        fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.' + subset + '.txt'))


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 


def compute_npmi_at_n(topics, ref_vocab, ref_counts, n=10):

    vocab_index = dict(zip(ref_vocab, range(len(ref_vocab))))
    n_docs, _ = ref_counts.shape

    npmi_means = []
    for topic in topics:
        words = topic
        npmi_vals = []
        for word_i, word1 in enumerate(words[:n]):
            if word1 in vocab_index:
                index1 = vocab_index[word1]
            else:
                index1 = None
            for word2 in words[word_i+1:n]:
                if word2 in vocab_index:
                    index2 = vocab_index[word2]
                else:
                    index2 = None
                if index1 is None or index2 is None:
                    npmi = 0.0
                else:
                    col1 = np.array(ref_counts[:, index1]>0, dtype=int)
                    col2 = np.array(ref_counts[:, index2]>0, dtype=int)
                    c1 = col1.sum()
                    c2 = col2.sum()
                    c12 = np.sum(col1 * col2)
                    if c12 == 0:
                        npmi = 0.0
                    else:
                        npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        # print(str(np.mean(npmi_vals)) + ': ' + ' '.join(words[:n]))
        npmi_means.append(np.mean(npmi_vals))
    print('npmi(n='+str(n)+'):', np.mean(npmi_means))
    return np.mean(npmi_means)


if __name__ == '__main__':
    main()
