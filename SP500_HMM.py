import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def algo(q, Y):
    # init
    p = 0.0
    fig, ax = plt.subplots()

    # Forward algorithm
    num_wk = Y.shape[0]
    emm_prob = np.array([[q, 1 - q], [1 - q, q]])
    trans_prob = np.array([[0.8, 0.2], [0.2, 0.8]])
    pi = np.array([0.2, 0.8])
    alpha = np.zeros((num_wk, 2))
    alpha[0, :] = pi * emm_prob[1, :]

    for wk in range(1, num_wk):
        if Y[wk] == 1:
            alpha[wk, 0] = emm_prob[0, 0] * (alpha[wk - 1, 0] * trans_prob[0, 0] + alpha[wk - 1, 1] * trans_prob[1, 0])
            alpha[wk, 1] = emm_prob[0, 1] * (alpha[wk - 1, 0] * trans_prob[0, 1] + alpha[wk - 1, 1] * trans_prob[1, 1])
        else:
            alpha[wk, 0] = emm_prob[1, 0] * (alpha[wk - 1, 0] * trans_prob[0, 0] + alpha[wk - 1, 1] * trans_prob[1, 0])
            alpha[wk, 1] = emm_prob[1, 1] * (alpha[wk - 1, 0] * trans_prob[0, 1] + alpha[wk - 1, 1] * trans_prob[1, 1])

    # Backward algorithm
    beta = np.zeros((num_wk, 2))
    beta[-1, :] = 1

    for wk in range(num_wk - 2, -1, -1):
        if Y[wk + 1] == 1:
            beta[wk, :] = trans_prob[:, 0] * emm_prob[0, 0] * beta[wk + 1, 0] + trans_prob[:, 1] * emm_prob[0, 1] * beta[wk + 1, 1]
        else:
            beta[wk, :] = trans_prob[:, 0] * emm_prob[1, 0] * beta[wk + 1, 0] + trans_prob[:, 1] * emm_prob[1, 1] * beta[wk + 1, 1]

    prob = alpha * beta / np.sum(alpha[num_wk - 1, :])
    p = prob[-1, 0]

    ax.plot(prob[:, 0])
    ax.set_title(f"q= {q}")
    plt.xlabel("Weeks")
    plt.ylabel("Probabilities")

    return p, fig

# main

mat = scipy.io.loadmat('sp500.mat')
mat = mat['price_move']

for q in [0.7, 0.9]:
    p, fig = algo(q, mat)
    fig.savefig('./'+str(q)+'.png')
    print('p: %.4f q: %.4f' % (p, q))
