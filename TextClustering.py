import numpy as np
import scipy.io
import scipy.io
import itertools
import numpy as np


# Accuracy of 82% achieved with random seed 20

# ======================== uncomment the following for extra task ========================
# n_topics = 5 # TODO specify num topics yourself
# cell = scipy.io.loadmat('nips.mat')
# mat = cell['raw_count'] # sparse mat of size (num_doc, num_words)
# wl = cell['wl']

# W = cluster_extra(mat, n_topics)

# display_topics(W, wl)

def cluster(T, K, num_iters = 1000, epsilon = 1e-12):
    """
    :param bow:
        bag-of-word matrix of (num_doc, V), where V is the vocabulary size
    :param K:
        number of topics
    :return:
        idx of size (num_doc), idx should be 1, 2, 3 or 4
    """
    # initialization
    np.random.seed(20)
    pi = np.random.random([K])
    mu = np.random.random([K, T.shape[1]])
    pi_og = np.zeros([K], dtype=float)
    iteration = 1 
    for i in range(K):
        mu[i] /= np.sum(mu[i])
    pi /= np.sum(pi)

    while np.linalg.norm(pi_og - pi) > epsilon:

        if iteration>num_iters:
            break

        # Expectation step
        gamma = np.zeros([T.shape[0], K], dtype=float)
        for d_i in range(T.shape[0]): # d_i : document i
            w_i = T[d_i, :] # w_i : word_i
            for c_i in range(K): # c_i : cluster i
                c = mu[c_i, :] # c : cluster
                gamma[d_i, c_i] = pi[c_i] * np.prod(np.power(c, w_i))
            gamma[d_i, :] /= np.sum(gamma[d_i, :])        
        pi_og = pi

        # Maximization step
        pi = np.mean(gamma, axis=0)
        mu = np.zeros([K, T.shape[1]])
        for i in range(K):
            for j in range(T.shape[1]):
                mu[i,j] = np.sum(np.multiply(gamma[:, i], T[:, j]))
            mu[i, :] /= np.sum(mu[i, :])
        iteration +=1
    idx = np.argmax(gamma, axis=1)
    return idx + 1


def acc_measure(idx):
    """

    :param idx:
        numpy array of (num_doc)
    :return:
        accuracy
    """

    mat = scipy.io.loadmat('data.mat')
    mat = mat['X']
    Y = mat[:, -1]

    # rotate for different idx assignments
    best_acc = 0
    for idx_order in itertools.permutations([1, 2, 3, 4]):

        for ind, label in enumerate(idx_order):
            Y[(ind)*100:(ind+1)*100] = label

        acc = (Y == idx).sum() / Y.shape[0]
        if acc > best_acc:
            best_acc = acc

    return best_acc


if __name__ == '__main__':
    acc_measure(np.array([1]*100 + [3]*100 + [2]*100 + [4]*100))



# main

mat = scipy.io.loadmat('data.mat')
mat = mat['X']
X = mat[:, :-1]

idx = cluster(X, 4)

acc = acc_measure(idx)

print('accuracy %.4f' % (acc))