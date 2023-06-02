import copy
import itertools
import multiprocessing as mp
import pickle

import numpy as np
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi

from qml import layers

class QcnnStruct:
    '''
    QCNN回路のベースとなるクラス
    '''

    Leyers = {layers.legacy_conv4_layer.name: layers.legacy_conv4_layer,
              layers.legacy_conv_layer.name: layers.legacy_conv_layer,
              layers.legacy_pool_layer.name: layers.legacy_pool_layer}
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.active_qubits = list(range(num_qubits))
        self.structure = []
        self.params = []

    def add_layer(self, layer, kwargs={}):
        '''
        structureのリストに引数で指定した種類とkeyword（start_index, label...)のタプルを追加
        '''
        self.structure.append((layer.name, kwargs))
        return
    
    def initialize_params(self, specific_params=None, random=False):
        '''
        回路の構成が決まったら、この関数を呼び出して回路にあった数のパラメータを初期化する。
        random=Trueにしたらrandomに設定され、それ以外なら1に初期化。paramsは[[layer1],[layer2]...]のように
        多重配列になっている。
        '''
        params = []
        for layer_info in self.structure:
            layer = self.Layers[layer_info[0]]
            layer_param_shape = layer.shape_params

            if random:
                layer_params = np.random.uniform(0, 2*np.pi, layer_param_shape)
            
            else:
                layer_params = np.ones(layer_param_shape)

            params.append(layer_params)

        self.params = params
        return
    

    @staticmethod
    def get_updated_active_qubits(active_qubits, group_len, target):
        '''
        active_qubitsのリストを更新する関数。
        acive_qubits = [1, 2, 3, 4, 5], group_len=3, target=2 の場合、updated_qubitsは
        [2, 5]になる。
        '''

        num_groups = len(active_qubits) // group_len
        update_qubits = []

        for i in range(num_groups):
            index = i * group_len + target
            update_qubits.append(active_qubits[index])

        return update_qubits
    

    def update_active_qubits(self, group_len, target):
        self.active_qubits = self.get_updated_active_qubits(self.active_qubits, group_len, target)
        return
    

    def reset_active_qubits(self):
        '''
        active_qubitsのインスタンス変数をリセットする関数
        '''
        self.active_qubits = list(range(self.num_qubits))
        return
    

    def generate_circ(self, params, draw=True):
        '''
        このクラスのコアメソッド
        今まで指定してきた、回路のstructureやparamに合わせて実際にcircuitを作成する関数
        '''
        circ = QuantumCircuit(self.num_qubits)

        for index, layer_info in enumerate(self.structure):
            layer_name, kwargs = layer_info
            layer = self.Layers[layer_name]

            layer.apply_layer(circ, params[index], self.active_qubits, kwargs)

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict[group_len]
                target = update_params_dict["target"]

                self.update_active_qubits(group_len, target)

        self.reset_active_qubits()  #一度回路をつくったらactive_qubitをリセットする
        circ = circ.reverse_bits()  #qiskit用に回路を反転する

        if draw:
            circ.draw(reverse_bits=True)

        return circ
    

    def get_final_state_active_qubits(self):
        '''
        conv, poolinglayerを複数かけた後、どのqubitがactiveなのか調べるメソッド
        '''
        active_qubits = self.active_qubits.copy()

        for index, layer_info in enumerate(self.structure):
            layer_name, kwargs = layer_info

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict[group_len]
                target = update_params_dict["target"]

                active_qubits = self.get_updated_active_qubits(active_qubits, group_len, target)

        return active_qubits
    
    
    @staticmethod
    def embedding(wf):
        '''
        複素数のリストを引数にとり、それを係数にした波動関数を生成する
        '''
        q_state = qi.Statevector(wf)
        return q_state
    
    @staticmethod
    def get_operator(circ):
        '''
        量子回路からオペレータ（行列表現）を作成するメソッド。
        '''
        operator = qi.Operator(circ)
        return operator
    

class Qcnn(QcnnStruct):
    '''
    QCNNを生成するのに使用するクラス
    '''
    def __init__(self, num_qubits):
        super().__init__(num_qubits)


    def forward(self, input_wfs, params):
        '''
        input_wfsを最初の状態として、qcnnを回す
        '''
        circ = self.generate_circ(params)
        predictions = np.zeros(len(input_wfs))

        for index, wf in enumerate(input_wfs):
            state = self.embedding(wf)
            state = state.evolve(circ)

            predictions[index] = self.middle_qubit_exp_value(state)

        return predictions
    

    def compute_grad(self, input_wfs, labels, epsilon=0.0001):
        '''
        微分をf'(x)|_a = f(a + epsilon) - f(a - epsilon) / 2*epsilon で計算する。
        ここでf(x)はパラメータをxとしたときの
        '''

        original_params = copy.deepcopy(self.params)
        gradient_mat = []

        for layer_index, layer_params in enumerate(self.params):
            layer_grad = np.zeros(len(layer_params))

            for param_index, _ in enumerate(layer_params):
                grad = 0
                for i in [1, -1]:
                    self.params[layer_index][param_index] += i * epsilon
                    grad += i * self.mse_loss(self.forward(input_wfs, self.params.copy()), labels)
                    self.params = copy.deepcopy(original_params)
                layer_grad[param_index] = grad / 2 * epsilon

            gradient_mat.append(layer_grad)

        return gradient_mat
    

    def pool_func_for_mp(self, indexes, input_wfs, labels, epsilon):
        '''
        paramsの中のindexesで指定した(i,j)番目のパラメータの変化率を計算する。これにより、マルチスレッディング
        (並列的に）勾配を計算できるようになる。
        '''
        i, j = indexes
        grad = 0

        for k in [1, -1]:
            params = copy.deepcopy(self.params)
            params[i][j] += k * epsilon
            grad += k * self.mse_loss(self.forward(input_wfs, params), labels)

        final_grad = grad / 2 * epsilon
        return i, j, final_grad
    
    def compute_grad_w_mp(self, input_wfs, labels, epsilon=0.0001):
        '''
        pool_func_for_mpを使ってパラメータ全体の勾配行列をマルチスレッディングで求めるモジュール
        '''
        indexes = [(i, j) for i, val in enumerate(self.params) for j, _ in enumerate(val)]

        p = mp.Pool(mp.cpu_count())
        grad_tuple = p.starmap(self.pool_func_for_mp, zip(indexes,
                                                          itertools.repeat(input_wfs),
                                                          itertools.repeat(labels),
                                                          itertools.repeat(epsilon)))
        gradient_mat = copy.deepcopy(self.params)
        for i, j, val in grad_tuple:
            gradient_mat[i][j] = val

        return gradient_mat
    
    def update_params(self, gradient_mat, learning_rate):
        '''
        gradient matrixを用いてlearning_rateに沿ってパラメータをアップデートする。
        '''
        for param_layer, grad_layer in zip(self.params, gradient_mat):
            param_layer -= learning_rate * grad_layer
        return
    
    def load_model(self, model_struct, specific_params):
        '''
        qcnnのlayer構成やパラメータを自分で指定し、インスタンス変数に格納することができる
        '''
        self.params = specific_params
        self.structure = model_struct
        return
    
    @staticmethod
    def mse_loss(predictions, label):
        '''
        平均二乗誤差を以下の式に従って計算
        mse = (len(x))^-1 * sum_i((x_label_i - x_pred_i)^2)
        '''
        num_entries = len(predictions)
        loss = np.sum(np.power((label - predictions), 2))

        return loss / num_entries
    
    def middle_qubit_exp_value(self, state_vect):
        '''
        
        '''
        final_active_qubits = self.get_final_state_active_qubits()
        middle_qubit = final_active_qubits[len(final_active_qubits) // 2]   #middle qubitのindex
        
        probability_vector = (np.abs(np.array(state_vect.data))) ** 2
        
        all_binary_combs = list(map(list, itertools.product([0,1], repeat=self.num_qubits)))
        new_list =np.array([elem for elem, val in enumerate(all_binary_combs) if val[middle_qubit] == 1])
        sums = np.sum(probability_vector[new_list])

        return (-1 * sums) + (1 * (1 - sums))
    
    @staticmethod
    def export_params(qcnn_struct, params, fname="model.pkl"):
        '''
        qcnnのstructureと学習済みのパラメータをバイト型に変換してpklファイルに保存
        '''
        with open(fname, 'wb') as file:
            pickle.dump((qcnn_struct, params), file)


    @staticmethod
    def import_params(fname="model.pkl"):
        '''
        pickleファイルを読み込み、structureとparametersの情報を抽出する
        '''
        with open(fname, 'rb') as file:
            qcnn_struct, params = pickle.load(file)

        return qcnn_struct, params
    
    def main():
        return
    
    if __name__ =="__main__":
        main()
