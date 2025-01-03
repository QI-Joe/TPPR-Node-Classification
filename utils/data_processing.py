from math import radians
import numpy as np
import random
import pandas as pd
import os
from utils.my_dataloader import data_load, Temporal_Splitting, Temporal_Dataloader
import torch

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, node_feat: np.ndarray = None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0
        self.node_feat = node_feat

  def sample(self,ratio):
    data_size=self.n_interactions
    sample_size=int(ratio*data_size)
    sample_inds=random.sample(range(data_size),sample_size)
    sample_inds=np.sort(sample_inds)
    sources=self.sources[sample_inds]
    destination=self.destinations[sample_inds]
    timestamps=self.timestamps[sample_inds]
    edge_idxs=self.edge_idxs[sample_inds]
    labels=self.labels[sample_inds]
    return Data(sources,destination,timestamps,edge_idxs,labels)


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []

  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]

    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0

    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
    
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)
  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y
    
    if np.any(graph.edge_attr != None):
        edge_attr = graph.edge_attr
    if np.any(graph.pos != None):
        pos = graph.pos
        pos = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    else:
        pos = graph.x

    TPPR_data = Data(sources= src, destinations=dest, timestamps=timestamp, edge_idxs = edge_idx, labels=labels, node_feat=pos)

    return TPPR_data

def get_data_TPPR(dataset_name, snapshot):
    r"""
    this function is used to convert the node features to the correct format
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method
    """
    graph, idx_list = data_load(dataset_name)
    non_split = True
    if snapshot<=3: 
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph_list = [graph]
    else:
        graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", \
                    snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
    
    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list)
    single_graph = False
    if lenth < 2: 
        lenth = 2
        single_graph = True

    for idxs in range(0, lenth-1):
        # covert Temproal_graph object to Data object
        items = graph_list[idxs]
        items.edge_attr = items.edge_attr # .numpy()
        # items.pos = items.pos.numpy()
        items.y = np.array(items.y)

        t_labels = items.y
        full_data = to_TPPR_Data(items)
        timestamp = full_data.timestamps
        val_time, test_time = np.quantile(timestamp, (0.20, 0.60))

        train_mask = timestamp <= val_time
        val_mask = (timestamp > val_time) & (val_time<=test_time)

        train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, full_data.node_feat)
        
        val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, full_data.node_feat)
        
        if single_graph:
            test_data = val_data
        else:
            test = graph_list[idxs+1]
            test_data = to_TPPR_Data(test)
        node_num = items.num_nodes
        node_edges = items.num_edges

        TPPR_list.append([full_data, train_data, val_data, test_data, node_num, node_edges])


    return TPPR_list

# path = "data/mooc/ml_mooc.npy"
# edge = np.load(path)
def load_feat(d):
    node_feats = None
    if os.path.exists('../data/{}/ml_{}_node.npy'.format(d,d)):
        node_feats = np.load('../data/{}/ml_{}_node.npy'.format(d,d)) 

    edge_feats = None
    if os.path.exists('../data/{}/ml_{}.npy'.format(d,d)):
        edge_feats = np.load('../data/{}/ml_{}.npy'.format(d,d))
    return node_feats, edge_feats


############## load a batch of training data ##############
def get_data(dataset_name):
  graph_df = pd.read_csv('data/{}/ml_{}.csv'.format(dataset_name,dataset_name))

  #edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
  #node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name)) 
  #node_features, edge_features = load_feat(dataset_name)

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  
  # ensure we get the same graph
  random.seed(2020)
  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)
  n_edges = len(sources)

  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * n_total_unique_nodes)))
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0


  # * the val set can indeed contain the new test node
  new_node_set = node_set - train_node_set
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

  test_mask = timestamps > test_time
  edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
  new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])
  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

  return full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data, n_total_unique_nodes, n_edges

