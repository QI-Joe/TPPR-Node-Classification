import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import sys
import torch.nn as nn
from utils.uselessCode import node_index_anchoring
from model.tgn_model import TGN

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size):

  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_acc = [], [], []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = data.n_interactions
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
 
    for batch_idx in range(num_test_batch):
      start_idx = batch_idx * TEST_BATCH_SIZE
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      sources_batch = data.sources[sample_inds]
      destinations_batch = data.destinations[sample_inds]
      timestamps_batch = data.timestamps[sample_inds]
      edge_idxs_batch = data.edge_idxs[sample_inds]


      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, n_neighbors, train = False)
      
      pos_prob=pos_prob.cpu().numpy() 
      neg_prob=neg_prob.cpu().numpy() 

      pred_score = np.concatenate([pos_prob, neg_prob])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])
      
      true_binary_label= np.zeros(size)
      pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1)

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_acc.append(accuracy_score(true_binary_label, pred_binary_label))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)



def eval_node_classification(tgn: TGN, decoder: LogRegression, val_src, val_edge_time, val_data):
  val_sample = np.array(list(set(val_src)))


  with torch.no_grad():
    decoder.eval()
    tgn.eval()

    val_emb = tgn.compute_node_probabilities(sources=val_src, edge_times=val_edge_time, train=False)
    val_pred: torch.Tensor = decoder.forward(val_emb)

  if isinstance(val_data.labels, list):
    val_data.labels = np.array(val_data.labels)
  val_label = val_data.labels[val_sample].reshape(-1,1)

  val_pred = val_pred.cpu().numpy()
  pred_prob = np.argmax(val_pred, axis=-1).reshape(-1,1)

  # auc_roc = roc_auc_score(val_label, pred_prob, multi_class="ovr")
  acc = accuracy_score(val_label, pred_prob)
  prec = average_precision_score(val_label, pred_prob)
  return acc, prec
