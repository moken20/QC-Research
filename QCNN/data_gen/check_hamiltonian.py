import numpy as np
import scipy.sparse as sparse

import hamiltonian as H


X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
I = np.array([[1, 0], [0, 1]])
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))


def test_dataset(n, dataset_file, tol):
    """
    Reads a datasets file and confirms the values for all lines (rows) are valid
    eigenvectors for the values of h1, h2, given at the beginning of the row.
    Uses np.eig to confirm, and is very slow at larger n values. This is just for
    testing to confirm our eigenvectors are valid energy states
    :param n: int - number of qbits
    :param dataset_file: str - location of file to check
    :param tol: float - absolute tolerance to accept
    :return:
    """
    H_Ham = H.Hamiltonian(n, (0, 1.6), (-1.6, 1.6), v=0)
    H_Ham.get_first_term()
    H_Ham.get_second_term()
    H_Ham.get_third_term()
    # H_Ham.calculate_terms()

    h1h2_ours, eigvecs = H.read_eigenvectors(dataset_file)
    print("Checking file ", dataset_file)
    misses = 0
    for i, (h1, h2) in enumerate(h1h2_ours):
        computed_hamiltonian = H_Ham.first_term + (H_Ham.second_term * h1) + (H_Ham.third_term * h2)

        eigenvalues, _ = H_Ham.find_eigval_with_np(computed_hamiltonian)
        test = (computed_hamiltonian @ eigvecs[i]) / eigenvalues
        real = np.array(eigvecs[i], dtype=complex)
        # print(misses)
        if not np.allclose(test, real, tol):
            misses += 1
    print("Total eigenvectors that are misses is:", misses)


def main():
    n = 9
    train_file = f"dataset_n={n}_train_dout.txt"
    test_file = f"dataset_n={n}_test.txt"
    tol = 0.01

    print('train----------------------')
    test_dataset(9, train_file, tol)
    # print('test--------------------')
    # test_dataset(9, test_file, tol)

if __name__ == '__main__':
    main()