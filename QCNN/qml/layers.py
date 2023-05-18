import numpy as np
import scipy.linalg as la
from qiskit import quantum_info as qi
from qiskit import QuantumCircuit


# Helper Functions ################################################
def b_mat(i, j, n):
    basis_matrix = np.zeros((n, n), dtype=np.float32)
    basis_matrix[i, j] = 1.0

    return basis_matrix


def generate_gell_mann(order):

    lst_of_gm_matricies = []
    for k in range(order):
        j = 0
        while j < k:
            sym_mat = b_mat(j, k, order) + b_mat(k, j, order)
            anti_sym_mat = complex(0.0, -1.0) * (b_mat(j, k, order) - b_mat(k, j, order))

            lst_of_gm_matricies.append(sym_mat), lst_of_gm_matricies.append(anti_sym_mat)
            j += 1

        if k < (order - 1):
            n = k + 1
            coeff = np.sqrt(2 / (n*(n+1)))

            sum_diag = b_mat(0, 0, order)
            for i in range(1, k+1):
                sum_diag += b_mat(i, i, order)

            diag_mat = coeff * (sum_diag - n*(b_mat(k+1, k+1, order)))
            lst_of_gm_matricies.append(diag_mat)

    return lst_of_gm_matricies


def get_conv_op(mats, parms):
    '''
    eの肩にgell mann行列とパラメータの1次結合が乗った行列をつくる
    '''
   
    final = np.zeros(mats[0].shape, dtype=np.complex128)
    for mat, parm in zip(mats, parms):  # sum over the gm matricies scaled by the parameters
        final += parm * mat

    return la.expm(complex(0, -1) * final)  # get the matrix exponential of the final matrix


def controlled_pool(mat):

    i_hat = np.array([[1.0, 0.0],
                      [0.0, 0.0]])
    j_hat = np.array([[0.0, 0.0],
                      [0.0, 1.0]])
    identity = i_hat + j_hat

    return np.kron(i_hat, identity) + np.kron(j_hat, mat)

def generate_uniformly_controlled_rotation(circ, params, control_qubit_indicies, target_qubit_index, axis='z', label=""):

    num_control_qubits = len(control_qubit_indicies)

    divisors = range(num_control_qubits - 1, -1, -1)
    divisors = [2**i for i in divisors]

    for iteration_num, theta in zip(range(1, 2**num_control_qubits+1), params):
        if axis == 'z':
            circ.rz(theta, target_qubit_index)
        elif axis == 'y':
            circ.ry(theta, target_qubit_index)
        else:
            circ.rx(theta, target_qubit_index)

        for divisor in divisors:
            if iteration_num % divisor == 0:
                control_element = int((num_control_qubits - 1) - np.log2(divisor))
                circ.cx(control_qubit_indicies[control_element], target_qubit_index)
                break

    print("--generate_uniformly_controlled_rotation------------------------")
    circ.draw(output='mpl')

    return


# Layer Implement ############################################################

def legacy_conv4_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    '''
    4qubitの畳み込み層を作成する

    kwargsはdict{"start_index" : __ , "label" : __ }
    で指定し、start_indexをqubit 0とすると、(2, 3)(0, 1)(0, 3)(0, 2)(1, 3)(1, 2)の順で4qubitの
    グループに2qubitのユニタリオペレータ（上のgell mannが肩にのったやつ）を作用させる
    その次はindexが+2 or +3されて上記のオペレータが同様にセットされる
    '''

    conv_operators = generate_gell_mann(4)  #2 qubit gell mann matrices
    u_conv = qi.Operator(get_conv_op(conv_operators, params))

    if "start_index" in kwargs: #layerの名称
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = '1c4'

    while index + 3 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]
        q_index_3 = active_qubits[index + 3]

        circ.unitary(u_conv, [q_index_2, q_index_3], label=label)
        circ.unitary(u_conv, [q_index, q_index_1], label=label)
        circ.unitary(u_conv, [q_index, q_index_3], label=label)
        circ.unitary(u_conv, [q_index, q_index_2], label=label)
        circ.unitary(u_conv, [q_index_1, q_index_3], label=label)
        circ.unitary(u_conv, [q_index_1, q_index_2], label=label)
        circ.barrier()

        if index == 0:
            index += 2
        else:
            index += 3

    if barrier:
        circ.barrier()

    print("4qubit畳み込み層追加後")
    circ.draw(output='mpl')

    return circ


def legacy_conv_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    '''
    一般化された畳み込み層の作成

    基本的には4qubitのときと同様だが、ユニタリオペレータが3qubitに作用する8x8のgell mannが
    肩にのった行列である。また、start_indexが0のとき、(0, 1, 2)のqubitにさようされ、その後
    active_qubitsの数だけ(3, 4, 5)(6, 7, 8)などに作用していく。
    '''

    conv_operators = generate_gell_mann(8)  #3 qubit operators
    u_conv = qi.Operator(get_conv_op(conv_operators, params))

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = 'ic'

    while index + 2 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]

        circ.unitary(u_conv, [q_index, q_index_1, q_index_2], label=label)
        index += 3

    if barrier:
        circ.barrier()

    return circ



