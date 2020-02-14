import numpy as np
import get_data
import random
import tensorflow as tf
import time
import copy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# ========================= preprocess circle info ===========================================
class DFSTreeNode:
    def __init__(self, node_idx=None, node_level=None):
        self.pre_node = None
        self.back_nodes = []
        self.next_nodes = []
        self.node_idx = node_idx
        self.node_level = node_level

# 返回一个 nodes_num*1 的向量，表示节点是否处于某个环中
def DFS(graph):
    flag_visited_nodes = np.zeros((graph.X.shape[0]))
    flag_avilable_edges = copy.copy(graph.A)
    nodes = {}    # node_idx: node objects
    nodes_in_circles = []
    edges_in_circles = []

    head_node = DFSTreeNode(node_idx=0, node_level=0)
    flag_visited_nodes[head_node.node_idx] = 1
    nodes[head_node.node_idx] = head_node

    current_node = head_node
    while(True):
        edge, edge_type = find_virgin_edge(current_node.node_idx,
                                           flag_avilable_edges[current_node.node_idx],
                                           flag_visited_nodes)
        if edge and edge_type=='forward_edge':
            # update flag_avilable_edges & flag_visited_nodes
            flag_avilable_edges[edge] = 0
            flag_avilable_edges[edge[1],edge[0]] = 0
            flag_visited_nodes[edge[1]] = 1
            # create next TreeNode
            next_node = DFSTreeNode(node_idx = edge[1], node_level = current_node.node_level+1)
            # set pointer
            next_node.pre_node = current_node
            current_node.next_nodes.append(next_node)
            nodes[next_node.node_idx] = next_node
            # info
            # update current_node
            current_node = next_node
        elif edge and edge_type=='back_edge':
            # update flag_avilable_edges
            flag_avilable_edges[edge[0],edge[1]] = 0
            flag_avilable_edges[edge[1],edge[0]] = 0
            # set pointer
            back_node = nodes[edge[1]]
            current_node.back_nodes.append(back_node)
            # find circles / find nodes in circles / find edges in circles
            nodes_in_c = find_nodes_in_circles(current_node, back_node)
            edges_in_c = find_edges_in_circles(current_node, back_node)
            for idx in nodes_in_c:
                if idx not in nodes_in_circles:
                    nodes_in_circles.append(idx)
            for edge in edges_in_c:
                if edge not in edges_in_circles:
                    edges_in_circles.append(edge)
        else:
            if current_node.pre_node:
                current_node = current_node.pre_node
            else:
                break

    np_nodes_in_circles = np.zeros([graph.X.shape[0],1])
    for idx in nodes_in_circles:
        np_nodes_in_circles[idx,0] = 1
    np_edges_in_circles = np.zeros(graph.A.shape)
    for edge in edges_in_circles:
        (idx1,idx2) = edge
        np_edges_in_circles[idx1,idx2] = 1
        np_edges_in_circles[idx2,idx1] = 1

    return np_nodes_in_circles, np_edges_in_circles


def find_virgin_edge(node_idx, avilable_edges_vector, flag_visited_nodes):
    neighbors_idx = np.where(avilable_edges_vector==1)[0]
    forward_edge_list = []
    back_edge_list = []
    for nb_idx in neighbors_idx:
        if flag_visited_nodes[nb_idx]==0:
            forward_edge_list.append((node_idx,nb_idx))
        else:
            back_edge_list.append((node_idx,nb_idx))
    if len(back_edge_list)>0:
        return back_edge_list[0], 'back_edge'
    elif len(forward_edge_list)>0:
        return forward_edge_list[0], 'forward_edge'
    else:
        return None, None

# 找到那些存在于某个环中的nodes
def find_nodes_in_circles(current_node, back_node):
    nodes_in_circles = []
    nodes_in_circles.append(back_node.node_idx)
    while(current_node.node_idx != back_node.node_idx):
        nodes_in_circles.append(current_node.node_idx)
        current_node = current_node.pre_node
    return nodes_in_circles

# 找到那些存在于某个环中的edges
def find_edges_in_circles(current_node, back_node):
    edges_in_circles = []
    edges_in_circles.append((current_node.node_idx, back_node.node_idx))
    while(current_node.node_idx != back_node.node_idx):
        edges_in_circles.append((current_node.node_idx, current_node.pre_node.node_idx))
        current_node = current_node.pre_node
    return edges_in_circles


# ======================== preprocess k-level aggregation ===================================================

def preprocess_k_aggregation(graph_set, k):
    x_dim = graph_set[0].X.shape[1]
    for i in range(k):
        if i==0:
            for graph in graph_set:
                graph.X_neighbor_k = np.matmul(graph.A, graph.X)
                graph.X = np.concatenate([graph.X, graph.X_neighbor_k], axis=1)
        else:
            for graph in graph_set:
                graph.X_neighbor_k_temp = np.matmul(graph.A, graph.X_neighbor_k[:,-x_dim:])
            for graph in graph_set:
                graph.X_neighbor_k = graph.X_neighbor_k_temp
                graph.X = np.concatenate([graph.X, graph.X_neighbor_k], axis=1)
    return graph_set

# =====================================================================

#================== SGN ===========================================================================

class SGN(tf.keras.Model):
    def __init__(self, graph_set):
        super(SGN, self).__init__()
        self.graph_set = graph_set
        self.input_size = self.graph_set[0].X.shape[1]
        self.output_size = self.graph_set[0].label.shape[1]
        self.batch_size = 40
        self.epoch = 100
        self.write_log = True
        self.k_fold = 5
        self.test_part = 0
        self.WEIGHTs = {}
        self.structure_size = 7
        self.query_num = 2
        self.cnn_kernel_num = 15
        self.graph_A_attr_size = 0 if self.graph_set[0].A_attr is None else self.graph_set[0].A_attr.shape[2]
        self.try_num = 5
        self.using_attention = True
        self.condition_factor = 'LSTM'
        self.cnn_layer_num = 2
        self.output_weight_Q = False

    def partition(self):
        self.test_proportion = 1/self.k_fold
        test_sample_num = int(len(self.graph_set)*self.test_proportion)
        test_set_start_index = self.test_part*test_sample_num
        test_set_end_index = min((self.test_part+1)*test_sample_num, len(self.graph_set))
        self.test_set = self.graph_set[test_set_start_index:test_set_end_index]
        self.train_set = self.graph_set[0:test_set_start_index] + self.graph_set[test_set_end_index:]

    def init_weights(self):
        self.WEIGHTs['q'] = {}
        for qn in range(self.query_num):
            self.WEIGHTs['q'][qn] = {}
            for i in range(self.structure_size):
                if self.condition_factor=='LSTM':
                    if i==0:
                        shape = (self.input_size, 1)
                    else:
                        shape = (self.input_size*2+self.graph_A_attr_size, 1)
                    self.WEIGHTs['q'][qn][i] = self.add_weight(shape=shape,
                                                initializer='truncated_normal',
                                                trainable=True)
                elif self.condition_factor=='concat':
                    if i==0:
                        shape = (self.input_size, 1)
                    else:
                        shape = (self.input_size*(i+1)+self.graph_A_attr_size, 1)
                    self.WEIGHTs['q'][qn][i] = self.add_weight(shape=shape,
                                                initializer='truncated_normal',
                                                trainable=True)
                elif self.condition_factor=='concat_simple' or self.condition_factor=='no':
                    if i==0:
                        shape = (self.input_size, 1)
                    else:
                        shape = (self.input_size+self.graph_A_attr_size, 1)
                    self.WEIGHTs['q'][qn][i] = self.add_weight(shape=shape,
                                                initializer='truncated_normal',
                                                trainable=True)

        if self.condition_factor=='LSTM':
            self._init_lstm_weights(input_size=self.input_size)

        if self.cnn_layer_num==1:
            self.cnn_kernel_shape = ((self.structure_size)* (self.structure_size)* (self.input_size*2+self.graph_A_attr_size),
                                 self.cnn_kernel_num)
            self.WEIGHTs['cnn'] = self.add_weight(shape=self.cnn_kernel_shape,
                                                initializer='truncated_normal',
                                                trainable=True)
        elif self.cnn_layer_num==2:
            self.WEIGHTs['cnn'] = {}
            self.cnn_kernel_shape = (3,3,self.input_size*2+self.graph_A_attr_size,self.cnn_kernel_num)
            self.WEIGHTs['cnn'][0] = self.add_weight(shape=self.cnn_kernel_shape,
                                                initializer='truncated_normal',
                                                trainable=True)
            self.cnn_kernel_shape = (5,5,self.cnn_kernel_num,self.cnn_kernel_num)
            self.WEIGHTs['cnn'][1] = self.add_weight(shape=self.cnn_kernel_shape,
                                                initializer='truncated_normal',
                                                trainable=True)

        self.WEIGHTs['mlp'] = self.add_weight(shape=(self.query_num*self.cnn_kernel_num, self.output_size),
                                              initializer='truncated_normal',
                                              trainable=True)

        if self.using_attention:
            self.WEIGHTs['att'] = self.add_weight(shape=(self.cnn_kernel_num, 1),
                                                initializer='truncated_normal',
                                                trainable=True)


    def _init_lstm_weights(self, input_size):
        self.WEIGHTs['lstm'] = {}
        self.WEIGHTs['lstm']['f'] = self.add_weight(shape=(input_size*2, input_size),
                                              initializer='truncated_normal',
                                              trainable=True)
        self.WEIGHTs['lstm']['i'] = self.add_weight(shape=(input_size*2, input_size),
                                              initializer='truncated_normal',
                                              trainable=True)
        self.WEIGHTs['lstm']['c'] = self.add_weight(shape=(input_size*2, input_size),
                                              initializer='truncated_normal',
                                              trainable=True)
        self.WEIGHTs['lstm']['o'] = self.add_weight(shape=(input_size*2, input_size),
                                              initializer='truncated_normal',
                                              trainable=True)

    def _get_lstm_state(self, input_data):
        # input_data shape = step_num*input_size
        step_num, input_size = input_data.shape
        init_state = np.zeros([1, input_size]).astype(np.float32)
        cell_state = init_state
        for sn in range(step_num):
            x = tf.concat([cell_state,input_data[sn:sn+1]], axis=1)
            f = tf.sigmoid(tf.matmul(x, self.WEIGHTs['lstm']['f']))
            i = tf.sigmoid(tf.matmul(x, self.WEIGHTs['lstm']['i']))
            c = tf.tanh(tf.matmul(x, self.WEIGHTs['lstm']['c']))
            o = tf.sigmoid(tf.matmul(x, self.WEIGHTs['lstm']['o']))
            cell_state = tf.multiply(cell_state, f)
            cell_state = tf.multiply(i, c) + cell_state
            hidden_state = tf.multiply(tf.tanh(cell_state), o)
        return hidden_state

    # generate structure DeepFirst
    def generate_structure(self, graph, weight):
        avilable_edge_A = copy.copy(graph.A)
        nodes_visited_order = {}
        accumulated_X = None
        query_X = None
        A_ = np.zeros((self.structure_size, self.structure_size)).astype(np.float32)
        if graph.A_attr is not None:
            A_attr_ = np.zeros((self.structure_size, self.structure_size, graph.A_attr.shape[2])).astype(np.float32)
        else:
            A_attr_ = None
        head_node = None
        current_node = None
        for i in range(self.structure_size):
            if accumulated_X is not None and self.Is_NaN(accumulated_X):
                print(accumulated_X)
            # 如果是选起始 node, score 是全部节点的得分;
            # 如果是选下一个 node, score 是邻居节点的得分 经未访问过的边
            if head_node is None:
                # score
                score = tf.matmul(graph.X, weight[0])
                score = (score-min(score))/(max(score)-min(score)+1e-9)
                att = tf.nn.softmax(score, axis=0)
                # rand_bias
                rand_bias = np.random.normal(loc=0.0,scale=1/6,size=score.shape).astype(np.float32)
                score_bias = score + rand_bias
                # found head node
                head_idx = tf.argmax(tf.reshape(score_bias,[-1]), axis=0).numpy()
                head_node = DFSTreeNode(node_idx=head_idx, node_level=1)
                #
                nodes_visited_order[head_idx] = 0
                accumulated_X = graph.X[head_idx:head_idx+1] * att[head_idx,0]
                query_X = graph.X[head_idx:head_idx+1]
                # update current position
                current_node = head_node
            else:
                while(max(avilable_edge_A[current_node.node_idx])==0 and current_node.pre_node):
                    current_node = current_node.pre_node
                if max(avilable_edge_A[current_node.node_idx])==1:
                    if graph.A_attr is not None:
                        current_node_e_attr = graph.A_attr[current_node.node_idx]
                        score_X = tf.concat([graph.X, current_node_e_attr], axis=1)
                    else:
                        score_X = graph.X
                    #
                    len_condition_X = current_node.node_level
                    w_q_idx = current_node.node_level
                    if self.condition_factor=='LSTM':
                        # X of query with LSTM
                        LSTM_input = query_X[0:len_condition_X]
                        state = self._get_lstm_state(LSTM_input)
                        state = tf.tile(tf.reshape(state, [1,-1]), [graph.X.shape[0],1])
                        score_input = tf.concat([state, score_X], axis=1)
                        # score
                        score = tf.matmul(score_input, weight[w_q_idx])
                    elif self.condition_factor=='concat':
                        # X of query
                        score_input = tf.reshape(query_X[0:len_condition_X], [1,-1])
                        score_input = tf.tile(score_input, [graph.X.shape[0],1])
                        score_input = tf.concat([score_input, score_X], axis=1)
                        # score
                        score = tf.matmul(score_input, weight[w_q_idx])
                    elif self.condition_factor=='concat_simple':
                        # condition X
                        score_input = tf.reshape(query_X[0:len_condition_X], [1,-1])
                        # condition weight
                        W_condition = [weight[step_i].numpy() for step_i in range(len_condition_X)]
                        W_condition = np.concatenate(W_condition, axis=0)
                        # condition bias
                        bias_condition = tf.matmul(score_input, W_condition)
                        # score
                        score = tf.matmul(score_X, weight[w_q_idx]) + bias_condition
                    elif self.condition_factor=='no':
                        # score
                        score = tf.matmul(score_X, weight[w_q_idx])
                    candidate_edges = avilable_edge_A[:,[current_node.node_idx]]
                    score = (score-min(score))/(max(score)-min(score)+1.0e-9)
                    score = tf.multiply(score + 1000, candidate_edges) - 1000
                    att = tf.nn.softmax(score, axis=0)
                    # rand_bias
                    rand_bias = np.random.normal(loc=0.0,scale=1/6,size=score.shape).astype(np.float32)
                    score_bias = score + rand_bias
                    # found next node
                    next_idx = tf.argmax(tf.reshape(score_bias,[-1]), axis=0).numpy()
                    next_node = DFSTreeNode(node_idx=next_idx, node_level=current_node.node_level+1)
                    #
                    next_node.pre_node = current_node
                    current_node.next_nodes.append(next_node)
                    nodes_visited_order[next_idx] = len(nodes_visited_order)
                    # update accumulated_X, query_X
                    accumulated_X = tf.concat([accumulated_X,
                                               graph.X[next_idx:next_idx+1] * att[next_idx,0]], axis=0)
                    query_X = tf.concat([query_X, graph.X[next_idx:next_idx+1]], axis=0)
                    # update current position
                    current_node = next_node
                    # add edges ***** # c_idx_, 后缀_表示 子结构 的意思
                    c_idx = current_node.node_idx
                    c_idx_ = nodes_visited_order[c_idx]
                    visited_nodes = list(nodes_visited_order.keys())
                    A_[c_idx_, 0:len(visited_nodes)] = graph.A[c_idx, visited_nodes]
                    A_[0:len(visited_nodes), c_idx_] = graph.A[visited_nodes, c_idx]
                    if graph.A_attr is not None:
                        A_attr_[c_idx_, 0:len(visited_nodes), :] = graph.A_attr[c_idx, visited_nodes, :]
                        A_attr_[0:len(visited_nodes), c_idx_, :] = graph.A_attr[visited_nodes, c_idx, :]
                    avilable_edge_A[c_idx, visited_nodes] = np.zeros([len(visited_nodes)])
                    avilable_edge_A[visited_nodes, c_idx] = np.zeros([len(visited_nodes)])
                else:
                    break

        # return accumulated_X & local_A
        if accumulated_X.shape[0]<self.structure_size:
            # if graph.X.shape[0]>accumulated_X.shape[0]:
            #     print('accumulated_X.shape[0]=%d,  graph.X.shape[0]=%d, self.structure_size=%d' %(
            #         accumulated_X.shape[0], graph.X.shape[0], self.structure_size))
            appending_size = self.structure_size - accumulated_X.shape[0]
            accumulated_X = tf.concat([accumulated_X, np.zeros((appending_size,graph.X.shape[1]))], axis=0)
        return accumulated_X, A_, A_attr_

    def CNN_module(self, accumulated_X, local_A, local_A_attr):
        X1 = accumulated_X
        X2 = accumulated_X
        X1 = tf.tile(tf.expand_dims(X1,0),[accumulated_X.shape[0],1,1])
        X2 = tf.tile(tf.expand_dims(X2,1),[1,accumulated_X.shape[0],1])
        X = tf.concat([X1,X2], axis=2)
        local_A = tf.tile(tf.expand_dims(local_A,2),[1,1,X.shape[2]])
        local_A = tf.multiply(X, local_A)
        if local_A_attr is not None:
            local_A = tf.concat([local_A, local_A_attr], axis=2)
        A__ = tf.reshape(local_A, [1,-1])
        Y = tf.matmul(A__,self.WEIGHTs['cnn'])
        Y = tf.keras.activations.tanh(Y)
        # Y has a shape of [1,self.cnn_kernel_num]
        return Y

    def CNN_module2(self, accumulated_X, local_A, local_A_attr):
        X1 = accumulated_X
        X2 = accumulated_X
        X1 = tf.tile(tf.expand_dims(X1,0),[accumulated_X.shape[0],1,1])
        X2 = tf.tile(tf.expand_dims(X2,1),[1,accumulated_X.shape[0],1])
        X = tf.concat([X1,X2], axis=2)
        local_A = tf.tile(tf.expand_dims(local_A,2),[1,1,X.shape[2]])
        local_A = tf.multiply(X, local_A)
        if local_A_attr is not None:
            local_A = tf.concat([local_A, local_A_attr], axis=2)
        local_A = tf.reshape(local_A, [1,self.structure_size,self.structure_size,-1])
        local_A = tf.nn.conv2d(local_A, self.WEIGHTs['cnn'][0],[1,1,1,1],padding='VALID')
        local_A = tf.keras.activations.relu(local_A)
        local_A = tf.nn.conv2d(local_A, self.WEIGHTs['cnn'][1],[1,1,1,1],padding='VALID')
        Y = tf.keras.activations.tanh(local_A)
        Y = tf.reshape(Y, [1, self.cnn_kernel_num])
        return Y


    # step1，同一个qurey规则下的结构，求和；step2，不同规则的representation应用MLP
    # step1，同一个query规则下的结构，attention求和；step2，不同规则的representation应用MLP
    def structures_learning(self, graph):
        if self.using_attention:
            representation = []
            for qn in range(self.query_num):
                repre_q = []
                for tn in range(self.try_num):
                    accumulated_X, local_A, local_A_attr = self.generate_structure(graph, self.WEIGHTs['q'][qn])
                    if self.cnn_layer_num==1:
                        rep = self.CNN_module(accumulated_X, local_A, local_A_attr)
                    elif self.cnn_layer_num==2:
                        rep = self.CNN_module2(accumulated_X, local_A, local_A_attr)
                    repre_q.append(rep)
                repre_q = tf.reshape(repre_q, [self.try_num, self.cnn_kernel_num])
                repre_att = tf.matmul(repre_q, self.WEIGHTs['att'])
                repre_att = tf.nn.softmax(repre_att, axis=0)
                repre_att = tf.tile(repre_att, [1,self.cnn_kernel_num])
                repre_q = tf.multiply(repre_q, repre_att)
                repre_q = tf.reduce_sum(repre_q, axis=0)
                representation.append(repre_q)
            representation = tf.reshape(representation, [1,-1])
            # MLP
            y = tf.matmul(representation, self.WEIGHTs['mlp'])
            y = tf.keras.activations.softmax(y)
            return y
        else:
            representation = []
            for qn in range(self.query_num):
                repre_q = []
                for tn in range(self.try_num):
                    accumulated_X, local_A, local_A_attr = self.generate_structure(graph, self.WEIGHTs['q'][qn])
                    if self.cnn_layer_num==1:
                        rep = self.CNN_module(accumulated_X, local_A, local_A_attr)
                    elif self.cnn_layer_num==2:
                        rep = self.CNN_module2(accumulated_X, local_A, local_A_attr)
                    repre_q.append(rep)
                repre_q = tf.reduce_sum(repre_q, axis=0)
                representation.append(repre_q)
            representation = tf.reshape(representation, [1,-1])
            # MLP
            y = tf.matmul(representation, self.WEIGHTs['mlp'])
            y = tf.keras.activations.softmax(y)
            return y

    def learing_a_batch(self, graph_set):
        Y_repr = []
        Y_real = []
        for graph in graph_set:
            y_repr = self.structures_learning(graph)
            y_real = graph.label
            Y_repr.append(y_repr)
            Y_real.append(y_real)
        Y_repr = tf.reshape(Y_repr, [-1, self.output_size])
        Y_real = tf.reshape(Y_real, [-1, self.output_size])
        return Y_real, Y_repr


    def calcu_acc(self, real_y, pred_y):
        index_real = tf.argmax(real_y, axis=1)
        index_pred = tf.argmax(pred_y, axis=1)
        acc_li = tf.cast(tf.equal(index_real, index_pred), tf.int32)
        acc = tf.reduce_sum(acc_li) / tf.shape(acc_li)[0]
        return acc.numpy()

    def train(self, optimizer=None, loss_computer=None, metrics=None):
        start_time = time.process_time()

        # output Q?
        if self.output_weight_Q:
            self._get_current_query_weights(-1)

        for epoch_i in range(self.epoch):
            batch_num = int(np.ceil(len(self.train_set)/self.batch_size))
            for batch_i in range(batch_num):
                s_index = batch_i * self.batch_size
                e_index = min((batch_i+1) * self.batch_size, len(self.train_set))
                this_batch = self.train_set[s_index:e_index]
                with tf.GradientTape() as tape:
                    real_Y, pred_Y = self.learing_a_batch(this_batch)
                    loss = loss_computer(real_Y, pred_Y)
                    acc = self.calcu_acc(real_Y, pred_Y)
                    auc = tf.keras.metrics.AUC()(real_Y, pred_Y)
                    grads = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    # 输出
                    string = '----- train loss acc auc:%f\t%f\t%f\tuse time %f' %(loss.numpy().mean(), acc, auc, time.process_time()-start_time)
                    print(string)
                    self._write_to_log(file_name=OUTPUT_FILE, string=string+'\n')
                    if batch_i>=batch_num-5:
                        # test_set
                        real_Y, pred_Y = self.learing_a_batch(self.test_set)
                        loss = loss_computer(real_Y, pred_Y)
                        acc = self.calcu_acc(real_Y, pred_Y)
                        auc = tf.keras.metrics.AUC()(real_Y, pred_Y)
                        # 输出
                        string = '      test  loss acc auc:%f\t%f\t%f' %(loss.numpy().mean(), acc, auc)
                        print(string)
                        self._write_to_log(file_name=OUTPUT_FILE, string=string+'\n')
                # 参数限制
            # output Q?
            if self.output_weight_Q:
                self._get_current_query_weights(epoch_i)
            # 输出 epoch & time
            string = '===== epoch %d finished, use time %f =============='%(epoch_i,time.process_time()-start_time)
            print(string)
            self._write_to_log(file_name=OUTPUT_FILE, string=string+'\n')


    def _write_to_log(self, file_name, string):
        if self.write_log:
            with open(file_name, 'a') as f:
                f.write(string)
    def _get_settings(self):
        string_settings = '\n==========================================================Model begin\n' + \
                        '*** query_num = '+str(self.query_num) + '\n' + \
                        '*** structure_size = '+str(self.structure_size) + '\n' + \
                        '*** condition_factor = '+str(self.condition_factor) + '\n' + \
                        '*** cnn_kernel_num = '+str(self.cnn_kernel_num) + '\n' + \
                        '*** cnn_kernel_shape = '+str(self.cnn_kernel_shape) + '\n' + \
                        '*** cnn_layer_num = '+str(self.cnn_layer_num) + '\n' + \
                        '*** try_num = '+str(self.try_num) + '\n' + \
                        '*** using_attention = '+str(self.using_attention) + '\n' + \
                        '*** test_part_k = '+str(self.test_part) + '\n' + \
                        '*** paras num = '+str(self._get_para_num()) + '\n' + \
                        '*** epoch num = '+str(self.epoch) + '\n' + \
                        '==========================================='
        return string_settings
    def _get_para_num(self):
        num_params = 0
        for v in self.trainable_variables:
            shape = tf.shape(v).numpy()
            num = 0
            for i in range(len(shape)):
                num = shape[i] if i==0 else num*shape[i]
            num_params += num
        return num_params
    def _get_current_query_weights(self, epoch_i):
        for qn in range(self.query_num):
            string = '========================= epoch %d done, weights as flows =======\n'%(epoch_i)
            for step_i in range(self.structure_size):
                string += str(self.WEIGHTs['q'][qn][step_i].numpy()) + '\n\n'
            self._write_to_log(file_name=WEIGHT_Q_FILE+'weight_q'+str(qn)+'_'+str(self.condition_factor)+'.txt', string=string)

    def run(self):
        self.init_weights()
        self.partition()
        string_settings = self._get_settings()
        print(string_settings)
        self._write_to_log(file_name=OUTPUT_FILE, string=string_settings+'\n')
        self.train(optimizer = tf.keras.optimizers.Adam(), loss_computer = tf.keras.losses.categorical_crossentropy)

    def Is_NaN(self, x):
        if tf.reduce_sum(x)>0 or tf.reduce_sum(x)<=0:
            return False
        else:
            return True


# ================== Main ==========================================================
graph_set = get_data.get_data_PTC()  # PROTEINS()
random.shuffle(graph_set)

for graph in graph_set:
    np_nodes_in_circles , np_edges_in_circles = DFS(graph)
    nodes_degree = np.reshape(np.sum(graph.A, axis=1), [-1,1])
    graph.X = np.concatenate([graph.X, np_nodes_in_circles, nodes_degree], axis=1).astype(np.float32)
#
preprocess_k_aggregation(graph_set, 2)


WEIGHT_Q_FILE = "XXX/"
OUTPUT_FILE = "XXX/results.txt"
model = SGN(graph_set)
model.batch_size = 40
model.epoch = 60
model.structure_size = 7
model.query_num = 2
model.condition_factor = 'no'
model.cnn_kernel_num = 15
model.try_num = 5
model.k_fold = 5
model.test_part = 0
model.using_attention = True
model.output_weight_Q = False

model.run()




