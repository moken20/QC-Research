import os
import time
import itertools
import numpy as np
import tqdm
import scipy.sparse as sparse
import scipy.sparse.linalg


def read_eigenvalues(file):
    with open(file, 'r+') as f:
        text_data = f.readlines()

        h_vals = []
        for i in range(len(text_data)):
            h1h2, eigenvector = text_data[i].split("_")

            h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
            text_data[i] = eigenvector

        return h_vals, np.loadtxt(text_data, dtype=complex)
    
def find_kron(array, index, q_bits):
    order = np.ones(q_bits)
    order[index-1] = 0
    assert index <= q_bits
    t = sparse.dia_matrix((pow(2, q_bits), pow(2, q_bits)), dtype=int)

    for i in range(1, len(order)):
        current = array if order[i] == 0 else II

        if i == 1:
            t = array if order[i-1] == 0 else II

        t = sparse.kron(t, current)

    return t.copy()



class Hamiltonian:
    def __init__(self,  qbits=4, h1_metadata=(0, 1.6), h2_metadata=(-1.6, -1.6), v=1):
        self.qbits = qbits
        self.verbose = v
        self.h1_min, self.h1_max = h1_metadata
        self.h2_min, self.h2_max = h2_metadata

        self.size = pow(2, self.qbits)
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)


    def get_first_term(self):
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.qbits -2):
            elem = i + 1
            if self.verbose: print(f"first term {elem}/{self.qbits - 2}")

            a = find_kron(Z, elem, self.qbits)
            b = find_kron(X, elem + 1, self.qbits)
            c = find_kron(Z, elem + 2, self.qbits)

            a_diag = a.diagonal()[..., None]
            c_diag = c.diagonal()[..., None]
            combined = b.multiply(a_diag).multiply(c_diag)

            self.first_term -= combined.toarray()


    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.qbits):
            elem = i + 1
            if self.verbose: print(f"second term{elem}/{self.qbits}")
            self.second_term -= find_kron(X, elem, self.qubits).toarray()


    def get_third_term(self):
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.qbits - 1):
            elem = i + 1
            if self.verbose: print(f"third term {elem}/{self.qbits - 1}")

            b1 = find_kron(X, elem, self.qbits)
            b2 = find_kron(X, elem+1, self.qbits)

            b1_rows, b1_cols = sparse.coo_matrix(b1, dtype=sparse.coo_matrix).nonzero()
            b2_rows, wrongly_ordered_b2_cols = b1_cols, []
            def extract_elem(elem):
                wrongly_ordered_b2_cols.append(elem)

            list(map(b2.getrow, filter(extract_elem, b2_rows)))

            b2_cols = []
            swaps = int(pow(2, self.qbits - 1 - elem))
            groups = [wrongly_ordered_b2_cols[i:i + swaps] for i in range(0, len(wrongly_ordered_b2_cols), swaps)]

            for j in range(int(len(groups) / 2)):
                switch = groups[2 * j:(2 * j) + 2]
                b2_cols.append([*switch[1], *switch[0]])

            b2_cols = list(itertools.chain(*b2_cols))

            combined = sparse.coo_matrix((np.ones(self.size, dtype=int), (b1_rows, b2_cols)),
                                         shape=(self.size, self.size))
            self.third_tem -= combined.toarray()

    
    def generate_data(self, h1_range, h2_range, name):
        t0 = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(f"{round(time.time() - t0, 4)}s elapsed to calculate term")

        filename = f'dataset_n={self.qbits}_' + name + ".txt"
        if os.path.isfile(filename): os.remove(filename)

        h1h2 = [[h1. h2] for h1 in np.linspace(self.h1_min, self.h1_max, h1_range)
                for h2 in np.linspace(self.h2_min, self.h2_max, h2_range)]
        
        for h1, h2 in tqdm.tqdm(h1h2):

            if name == "train": h2 = 0

            h = self.first_term + (self.second_term * h1) + (self.third_term * h2)
            eigenvalue, eigenvector = self.find_eigval_with_sparse(h)


            with open(filename, 'a+') as f:
                f.write(f"{h1, h2}_")
                for line in eigenvector:
                    f.write(str(line) + " ")
                f.write("\n")

    
    @staticmethod
    def find_eigval_with_sparse(h):
        b, c = sparse.linalg.eigs(h, k=1, which='SR', tol=1e-16)
        return b, c.flatten()
    
    @staticmethod
    def find_eigval_with_np(h):
        ww, vv = np.linalg.eig(h)
        index = np.where(ww == np.amin(ww))
        np_eig_vec, np_eig_val = vv[:, index], ww[index]

        eig_vect_list = []
        for eigVal in range(len(np_eig_val)):
            temp_vec = []

            for eigVec in range(len(np_eig_vec)):
                temp_vec.append(np_eig_vec[eigVec][0][eigVal])
            eig_vect_list.append(np.array(temp_vec))

        sum_vec = np.sum(eig_vect_list, axis=0)
        return np_eig_val[0], sum_vec / np.linalg.norm(sum_vec)
    

    def test_datast(self, h, possible_eigenvalue):
        _, np_eig_vec = self.find_eigval_with_np(h)
        magnitude = (h @ np_eig_vec) / possible_eigenvalue

        assert np.allclose(magnitude, np.array(np_eig_vec, dtype=complex), 1e-9)

    
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))
Z = sparse.dia_matrix((np.array([1, -1]), np.array([0])), dtype=int, shape=(2, 2))
X = sparse.dia_matrix((np.array([np.ones(1)]), np.array([-1])), dtype=int, shape=(2, 2))
X.setdiag(np.ones(1), 1)

if __name__ == '__mian__':
    s = time.time()
    n = 9
    
    H = Hamiltonian(n)
    H.generate_data(40, 1, "train")
    H.generate_data(64, 64, "test")

    print(f"Time for creating dataset was {time.time() - s} seconds")