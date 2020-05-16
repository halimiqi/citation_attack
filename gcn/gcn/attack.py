import numpy as np
import os
import pickle
import scipy.sparse as sp
import datetime
from gcn.utils import *


## attack 的时候应该直接攻击test集的50000个节点， 不要用训练集的了，没用
def add_node_feature():
    ## read vectors
    number_of_target = 500
    number_perturb_edge = 100
    repeat_num = 10
    threshold = 50
    num_target_after_repeat = number_of_target / repeat_num
    embeddings = []
    select_node_fun = "find_weak_point"
    with open(os.path.join("./hidden_vectors", "16_200502160359.pkl"), "rb") as f:
        embeddings = pickle.load(f)
    last_vec = embeddings[0]
    outputs = embeddings[1]    # we use this as the target ones.
    target_embeds = outputs
    ## the softmax of the outputs
    target_embeds_soft = softmax(target_embeds.T)
    target_embeds_soft = target_embeds_soft.T

    ## find target_node
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_kdd("kdd")
    if select_node_fun == "find_min_degree":
        target_indexes = find_min_degree(adj, int(num_target_after_repeat), threshold)
    elif select_node_fun == "find_max_degree":
        target_indexes = find_max_degree(adj, int(num_target_after_repeat), threshold)
    elif select_node_fun == "find_weak_point":
        target_indexes = find_weak_point(adj, int(num_target_after_repeat), target_embeds_soft)
    selected_embeds = target_embeds_soft[target_indexes,:]
    ## get the target embeds on the test set
    target_embeds_test = target_embeds_soft[543486:,:]
    ## calculate similarity l2 norm
    score = []
    for i in range(len(selected_embeds)):
        temp_score = np.sqrt(np.sum((selected_embeds[i,:] - target_embeds_test)**2, axis = -1))
        score.append(temp_score)
    #score = selected_embeds.dot(last_vec.T)  # the shape is 500 * 50000
    score = np.array(score)
    #indexes = np.argsort(score, axis=1)
    selected_indexes = np.argsort(-score, axis=1)
    selected_indexes = selected_indexes[:,:repeat_num*(number_perturb_edge-1)] # shape 500*99
    ## add the move part of the
    selected_indexes = selected_indexes + 543486
    #coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=np.int)
    adj_coo = adj.tocoo()
    rows = list(adj_coo.row)
    cols = list(adj_coo.col)
    data = list(adj_coo.data)
    for i, idx in enumerate(target_indexes):
        # connect target
        for j in range(repeat_num):
            rows.extend([idx, adj.shape[0]+i*repeat_num + j])
            cols.extend([adj.shape[0]+i*repeat_num + j, idx])
            data.extend([1,1])
            # connect others
            rows.extend((number_perturb_edge-1)*[adj.shape[0]+i*repeat_num + j])
            #cols.extend(selected_indexes[i,:])
            cols.extend(selected_indexes[i,i*repeat_num:i*repeat_num+number_perturb_edge-1])
            data.extend((number_perturb_edge-1)*[1])
            #rows.extend(selected_indexes[i, :])
            rows.extend(selected_indexes[i, i*repeat_num:i*repeat_num+number_perturb_edge-1])
            cols.extend((number_perturb_edge-1) * [adj.shape[0]+i*repeat_num + j])
            data.extend((number_perturb_edge-1) * [1])
    ## form the new adj
    new_adj = sp.coo_matrix((data, (rows, cols)), shape=(adj.shape[0]+number_of_target, adj.shape[0]+number_of_target), dtype=np.int)
    new_adj = new_adj.tocsr()
    ## features
    new_features = []
    for i, idx in enumerate(target_indexes):
        for j in range(repeat_num):
            target_feature = features[idx,:]
            res_features = features[selected_indexes[i],:]
            res_feature_mean = np.mean(res_features,axis = 0)
            new_feature = (target_feature + res_feature_mean) / 2
            new_features.append(new_feature)
            sub_adj = new_adj[adj.shape[0]:adj.shape[0]+number_of_target, :]
    return sub_adj, np.array(new_features)



def compare():

    return

def find_max_degree(adj, number,threshold):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum = rowsum[:,0]
    ori_index = np.arange(len(rowsum))
    sel_index = ori_index[rowsum < threshold]
    #sel_rowsum = rowsum[rowsum < threshold]
    ## get the test index
    sel_index = sel_index[sel_index>543485]
    sel_rowsum = rowsum[sel_index]
    ## get the maximum index
    max_indexes = np.argsort(-sel_rowsum)
    target_indexes = sel_index[max_indexes[:number]]
    return target_indexes

def find_min_degree(adj, number ,threshold):
    """
    the threshold is the target node degree from where
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum = rowsum[:, 0]
    ori_index = np.arange(len(rowsum))
    sel_index = ori_index[rowsum> threshold]
    #sel_rowsum = rowsum[rowsum>threshold]
    ## get the test index
    sel_index = sel_index[sel_index>543485]
    sel_rowsum = rowsum[sel_index]
    # get the min index
    min_indexes = np.argsort(sel_rowsum)
    ## this is the target number of the future things
    target_indexes = sel_index[min_indexes[:number]]
    return target_indexes

def find_weak_point(adj, number,  embeds):
    ori_index = np.arange(adj.shape[0])
    sel_index = ori_index[ori_index>543485]
    sel_embeds = embeds[sel_index]
    score = np.max(sel_embeds, axis=1)
    weak_indexes = np.argsort(score)
    target_indexes = sel_index[weak_indexes[:number]]
    return target_indexes
def main():
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    os.mkdir(os.path.join("./results/", current_time))
    new_adj, new_features = add_node_feature()
    ## save the adj.pkl
    with open(os.path.join("./results/",current_time, "adj.pkl"), "wb") as f:
        pickle.dump(new_adj, f)
    ## save the feature.npy
    np.save(os.path.join("./results/",current_time, "feature.npy"), new_features)
if __name__ == "__main__":
    main()