import numpy as np
import scipy.io
import time
import numpy as np


np.random.seed(10)

def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    max_iter = 5
    learning_rate = 6e-3
    reg_coef = 2e-2
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    for i in range(max_iter):
        for j in range(n_user):
            for k in range(n_item):
                if rate_mat[j, k] > 0:
                    e_i_j = rate_mat[j, k] - np.dot(U[j], V[k])
                    for l in range(lr):
                        if with_reg:
                            U[j, l] += learning_rate * (2 * e_i_j * V[k, l] - reg_coef * U[j, l])
                            V[k, l] += learning_rate * (2 * e_i_j * U[j, l] - reg_coef * V[k, l])
                        else:
                            U[j, l] += learning_rate * (2 * e_i_j * V[k, l])
                            V[k, l] += learning_rate * (2 * e_i_j * U[j, l])
    return U, V


# Main
def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)


cell = scipy.io.loadmat('movie_data.mat')
rate_mat = cell['train']
test_mat = cell['test']

low_rank_ls = [1, 3, 5]
for lr in low_rank_ls:
    for reg_flag in [False, True]:
        st = time.time()
        U, V = my_recommender(rate_mat, lr, reg_flag)

        t = time.time() - st

        print('SVD-%s-%i\t%.4f\t%.4f\t%.2f\n' % ('withReg' if reg_flag else 'noReg', lr,
                                                 rmse(U, V, rate_mat), rmse(U, V, test_mat), t))