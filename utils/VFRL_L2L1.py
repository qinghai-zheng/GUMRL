# This file defines the learning process of VFRL

import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance


class VFRL:

    def __init__(self, k=10, measure='cosine', eps=1e-5, verbose=False):

        self.k = k
        self.measure = measure
        self.eps = eps
        self.verbose = verbose
        self.U = None
        self.i = None
        self.j = None
        self.n_samples = None

    @staticmethod
    def half_quadratic_function(data, mu):

        return (1 / (mu + np.sum(data**2, axis=1)))**(0.5)

    def compute_obj(self, X, U, G, lpq, i, j, lambda_, beta, mu, weights, iter_num):

        data = 0.5 * np.sum(np.sum((X - U)**2))
        diff = np.sum((U[i, :] - U[j, :])**2, axis=1)
        smooth = lambda_ * 0.5 * (np.inner(lpq*weights, diff) +
                                  np.inner(weights, mu * lpq-1))

        LX_mse = 0.5 * beta * np.sum(np.sum((U - G)**2))

        obj = data + smooth + LX_mse
        if self.verbose:
            print(' {} | {} | {} | {}'.format(iter_num, data, smooth, obj))

        return obj

    @staticmethod
    def m_knn(X, k, measure='cosine'):
        """
        This code is taken from:
        https://bitbucket.org/sohilas/robust-continuous-clustering/src/
        The original terms of the license apply.
        """

        samples = X.shape[0]
        batch_size = 10000
        b = np.arange(k+1)
        b = tuple(b[1:].ravel())

        z = np.zeros((samples, k))
        weigh = np.zeros_like(z)

        for x in np.arange(0, samples, batch_size):
            start = x
            end = min(x+batch_size, samples)

            w = distance.cdist(X[start:end], X, measure)

            y = np.argpartition(w, b, axis=1)

            z[start:end, :] = y[:, 1:k + 1]
            weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end-start), k)),
                                               tuple(y[:, 1:k+1].ravel())], (end-start, k))
            del w

        ind = np.repeat(np.arange(samples), k)

        P = csr_matrix((np.ones((samples*k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

        Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
        P = triu(P, k=1)

        V = np.asarray(find(P)).T
        return V[:, :2].astype(np.int32)

    def run_vfrl(self, X, G, beta, w, max_iter=100, inner_iter=4):

        X = X.astype(np.float32)

        w = w.astype(np.int32)
        assert w.shape[1] == 2

        i = w[:, 0]
        j = w[:, 1]

        n_samples, n_features = X.shape

        n_pairs = w.shape[0]

        xi = np.linalg.norm(X, 2)

        R = scipy.sparse.coo_matrix((np.ones((i.shape[0]*2,)),
                                     (np.concatenate([i, j], axis=0),
                                      np.concatenate([j, i], axis=0))), shape=[n_samples, n_samples])

        n_conn = np.sum(R, axis=1)

        n_conn = np.asarray(n_conn)

        weights = np.mean(n_conn) / np.sqrt(n_conn[i]*n_conn[j])
        weights = weights[:, 0]

        U = X.copy()

        lpq = np.ones((i.shape[0],))

        epsilon = np.sqrt(np.sum((X[i, :] - X[j, :])**2 + self.eps, axis=1))

        epsilon[epsilon/np.sqrt(n_features) < 1e-2] = np.max(epsilon)

        epsilon = np.sort(epsilon)

        mu = 3.0 * epsilon[-1]**2

        top_samples = np.minimum(250.0, math.ceil(n_pairs*0.01))

        delta = np.mean(epsilon[:int(top_samples)])
        epsilon = np.mean(epsilon[:int(math.ceil(n_pairs*0.01))])

        R = scipy.sparse.coo_matrix((np.concatenate([weights*lpq, weights*lpq], axis=0),
                                     (np.concatenate([i,j],axis=0), np.concatenate([j,i],axis=0))),
                                    shape=[n_samples, n_samples])

        D = scipy.sparse.coo_matrix((np.squeeze(np.asarray(np.sum(R,axis=1))),
                                     ((range(n_samples), range(n_samples)))),
                                    (n_samples, n_samples))

        eigval = scipy.sparse.linalg.eigs(D-R, k=1, return_eigenvectors=False).real

        lambda_ = xi / eigval[0]

        if self.verbose:
            print('mu = {}, lambda = {}, epsilon = {}, delta = {}'.format(mu, lambda_, epsilon, delta))
            print(' Iter | Data \t | Smooth \t | Obj \t')    

        obj = np.zeros((max_iter,))

        inner_iter_count = 0

        for iter_num in range(1, max_iter):

            lpq = self.half_quadratic_function(U[i, :] - U[j, :], mu)

            obj[iter_num] = self.compute_obj(X, U, G, lpq, i, j, lambda_,beta, mu, weights, iter_num)

            R = scipy.sparse.coo_matrix((np.concatenate([weights*lpq, weights*lpq], axis=0),
                                         (np.concatenate([i,j],axis=0), np.concatenate([j,i],axis=0))),
                                        shape=[n_samples, n_samples])

            D = scipy.sparse.coo_matrix((np.asarray(np.sum(R, axis=1))[:, 0], ((range(n_samples), range(n_samples)))),
                                        shape=(n_samples, n_samples))

            M = scipy.sparse.eye(n_samples) + lambda_ * (D-R)/(1+beta)

            U = scipy.sparse.linalg.spsolve(M, (X+beta*G)/(1+beta))

            inner_iter_count += 1

            if (abs(obj[iter_num-1]-obj[iter_num]) < 1e-1) or inner_iter_count == inner_iter:
                if mu >= delta:
                    mu /= 2.0
                elif inner_iter_count == inner_iter:
                    mu = 0.5 * delta
                else:
                    break

                eigval = scipy.sparse.linalg.eigs((D-R)/(1+beta), k=1, return_eigenvectors=False).real
                xi = np.linalg.norm((X+beta*G)/(1+beta), 2)
                lambda_ = xi / eigval[0]
                inner_iter_count = 0

        self.U = U.copy()
        self.i = i
        self.j = j
        self.n_samples = n_samples

        return U

    def fit(self, X, beta, G):

        assert type(X) == np.ndarray
        assert len(X.shape) == 2

        mknn_matrix = self.m_knn(X, self.k, measure=self.measure)

        U = self.run_vfrl(X, G, beta, mknn_matrix)

        return U