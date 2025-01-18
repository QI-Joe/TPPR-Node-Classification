import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction, eval_node_classification, LogRegression
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, Running_Permit
from utils.data_processing import get_data_TPPR
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
from utils.my_dataloader import to_cuda, Temporal_Splitting, Temporal_Dataloader, data_load
from itertools import chain
from utils.uselessCode import node_index_anchoring

import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)


parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='cora')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=7, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.3, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="diffusion", help='Type of embedding module')

parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--save_best',action='store_true', help='store the largest model')
parser.add_argument('--tppr_strategy', type=str, help='[streaming|pruning]', default='streaming')
parser.add_argument('--topk', type=int, default=20, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')


parser.add_argument('--ignore_edge_feats', action='store_true')
parser.add_argument('--ignore_node_feats', action='store_true')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')

# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy streaming --gpu 0 --alpha_list 0.1 --beta_list 0.9

args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = True
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

if args.save_best:  
  best_checkpoint_path = f'./saved_checkpoints/{args.data}-{args.n_epoch}-{args.lr}-{args.tppr_strategy}-{str(args.alpha_list)}-{str(args.beta_list)}-{args.topk}.pth'
else:
  best_checkpoint_path = f'./saved_checkpoints/{time.time()}.pth'

print(best_checkpoint_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not args.enable_random:
  torch.manual_seed(0)
  np.random.seed(0)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


#################### get filename here #####################
filename=args.data
tppr_strategy=args.tppr_strategy
if tppr_strategy!='None':
  args.embedding_module='diffusion'
  filename=filename+'_'+tppr_strategy
  filename=filename+'_topk_'+str(args.topk)
  filename=filename+'_alpha_'+str(args.alpha_list)
  filename=filename+'_beta_'+str(args.beta_list)
  if tppr_strategy=='pruning':
    filename=filename+'_width_'+str(args.n_degree)+'_depth_'+str(args.n_layer)
filename=filename+'_bs_'+str(BATCH_SIZE)+'_layer_'+str(args.n_layer)+'_epoch_'+str(args.n_epoch)+'_lr_'+str(args.lr)    
if args.enable_random:
  filename=filename+'_random_seed'
print(filename)

######################## get logger ########################
Path(f"log/{args.data}").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f'log/{args.data}/{filename}')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

round_list = get_data_TPPR(DATA, snapshot=args.n_runs)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
training_strategy = "node"
NODE_DIM = round_list[0][0].node_feat.shape[1]

all_run_times = time.time()
for i in range(3):

  full_data, train_data, val_data, test_data, n_nodes, n_edges = round_list[i]
  num_classes = np.max(full_data.labels)+1

  args.n_nodes = n_nodes +1
  args.n_edges = n_edges +1

  edge_feats = None
  node_feats = full_data.node_feat
  node_feat_dims = full_data.node_feat.shape[1]

  if edge_feats is None or args.ignore_edge_feats: 
    print('>>> Ignore edge features')
    edge_feats = np.zeros((args.n_edges, 1))
    edge_feat_dims = 1

  train_ngh_finder = get_neighbor_finder(train_data)
  full_ngh_finder = get_neighbor_finder(full_data)
  # train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  # val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  # test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_feats, edge_features=edge_feats, device=device,
            n_layers=NUM_LAYER,n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            node_dimension = NODE_DIM, time_dimension = TIME_DIM, memory_dimension=NODE_DIM,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            args=args)
  
  projector = LogRegression(in_channels=128*3, num_classes=num_classes).to(device)
  # decoder = LogRegression(in_channels=node_feat_dims*3, num_classes=num_classes).to(device)

  criterion = torch.nn.BCELoss()
  criterion_node = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = torch.optim.Adam(chain(tgn.parameters(), projector.parameters()), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  t_total_epoch_train=0
  t_total_epoch_val=0
  t_total_epoch_test=0
  t_total_tppr=0
  stop_epoch=-1

  embedding_module = tgn.embedding_module

  print(f"the embedding module is {tgn.embedding_module_type}")

  train_src = np.concatenate([train_data.sources, train_data.destinations])
  timestamps_train = np.concatenate([train_data.timestamps, train_data.timestamps])

  embedding_module.streaming_topk_node(source_nodes=train_src, timestamps=timestamps_train, edge_idxs=train_data.edge_idxs)

  train_tppr_time=[]
  tppr_filled = False

  for epoch in range(NUM_EPOCH):
    t_epoch_train_start = time.time()
    tgn.reset_timer()
    num_instance = len(train_data.sources)
    BATCH_SIZE = num_instance
    num_batch = math.ceil(num_instance/BATCH_SIZE)

    train_ap=[]
    train_acc=[]
    train_auc=[]
    train_loss=[]

    tgn.memory.__init_memory__()
    # if args.tppr_strategy=='streaming':
    #   tgn.embedding_module.reset_tppr()
    tgn.set_neighbor_finder(train_ngh_finder)

    # model training
    tgn = tgn.train()
    optimizer.zero_grad()

    node_emb = tgn.compute_node_probabilities(NUM_NEIGHBORS, train=True)
    node_emb = projector.forward(node_emb)
    
    labels = train_data.labels
    labels_on_GPU = torch.tensor(labels).to(device)
    loss = criterion_node(node_emb, labels_on_GPU)
    
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

    with torch.no_grad():
      node_pred = node_emb.argmax(-1).cpu().numpy()
      train_ap.append(average_precision_score(labels.reshape(-1,1), node_pred.reshape(-1,1)))
      train_acc.append(accuracy_score(labels, node_pred))
      print(f"(TPPR) | epoch {epoch} train ACC {train_acc[-1]:.5f}, train AP {train_ap[-1]:.5f}, Loss: {loss.item():.4f}")

    if (epoch+1) % 50 == 0:
      epoch_tppr_time = tgn.embedding_module.t_tppr
      train_tppr_time.append(epoch_tppr_time)

      epoch_train_time = time.time() - t_epoch_train_start
      t_total_epoch_train+=epoch_train_time
      train_ap=np.mean(train_ap)
      # train_auc=np.mean(train_auc)
      train_acc=np.mean(train_acc)
      train_loss=np.mean(train_loss)

      # change the tppr finder to validation and test
      if args.tppr_strategy=='streaming':
        tgn.embedding_module.reset_tppr()
        tgn.embedding_module.fill_tppr(train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs, tppr_filled)
        tppr_filled = True
      tgn.set_neighbor_finder(full_ngh_finder)

      ########################  Model Validation on the Val Dataset #######################
      t_epoch_val_start=time.time()
      ### transductive val
      train_memory_backup = tgn.memory.backup_memory()
      if args.tppr_strategy=='streaming':
        train_tppr_backup = tgn.embedding_module.backup_tppr()

      val_anchor_acc, val_acc, val_ap = eval_node_classification(tgn=tgn, decoder = projector, data = val_data, num_classes=num_classes, n_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)
      # val_ap, val_auc, val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE)

      val_memory_backup = tgn.memory.backup_memory()
      if args.tppr_strategy=='streaming':
        val_tppr_backup = tgn.embedding_module.backup_tppr()
      tgn.memory.restore_memory(train_memory_backup)
      if args.tppr_strategy=='streaming':
        tgn.embedding_module.restore_tppr(train_tppr_backup)

      ### inductive val
      # nn_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=new_node_val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE)
      # tgn.memory.restore_memory(val_memory_backup)
      # if args.tppr_strategy=='streaming':
      #   tgn.embedding_module.restore_tppr(val_tppr_backup)


      epoch_val_time = time.time() - t_epoch_val_start
      t_total_epoch_val += epoch_val_time
      epoch_id = epoch+1
      # logger.info('epoch running time: {}, tppr: {:.4f}, train: {:.4f}, val: {:.4f}'.format(epoch_id, epoch_tppr_time, epoch_train_time, epoch_val_time))
      # logger.info('train auc: {:.4f}, train ap: {:.4f}, train acc: {:.4f}, train loss: {:.4f}'.format(train_auc, train_ap, train_acc, train_loss))
      print('epoch {} val ap: {:.4f}'.format(epoch, val_ap))
      print('(TPPR) | val acc: {:.4f}, val ancor acc {:.4f}'.format(val_acc, val_anchor_acc))

    """
    last_best_epoch=early_stopper.best_epoch
    if early_stopper.early_stop_check(val_ap):
      stop_epoch=epoch_id
      if False:
        model_parameters,tgn.memory=torch.load(best_checkpoint_path)
        tgn.load_state_dict(model_parameters)
        tgn.eval()
      break
    else:
      if epoch==early_stopper.best_epoch:
        ...
        # torch.save((tgn.state_dict(),tgn.memory), best_checkpoint_path)
    """
  sys.exit(0)
  ######################  Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()

  ### transductive test
  val_memory_backup = tgn.memory.backup_memory()
  if args.tppr_strategy=='streaming':
    val_tppr_backup = tgn.embedding_module.backup_tppr()

  test_ap, test_auc, test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=test_rand_sampler,data=test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE)

  tgn.memory.restore_memory(val_memory_backup)
  if args.tppr_strategy=='streaming':
    tgn.embedding_module.restore_tppr(val_tppr_backup)

  ### inductive test
  # nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler= nn_test_rand_sampler, data=new_node_test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE)
  t_test=time.time()-t_test_start

  train_tppr_time=np.array(train_tppr_time)[1:]
  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH
  logger.info(f'### num_epoch {NUM_EPOCH}, epoch_train {round(t_total_epoch_train/NUM_EPOCH, 4)}, epoch_val {round(t_total_epoch_val/NUM_EPOCH, 4)}, epoch_test {round(t_test, 4)}, train_tppr {round(np.mean(train_tppr_time), 4)}')
  logger.info(f"### all epoch train time {round(t_total_epoch_train, 4)}, entire tppr finder time {round(np.sum(train_tppr_time), 4)}, entire run time without data loading: {round(time.time()-all_run_times, 4)}")
  
  logger.info('Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))

  if not args.save_best:
    os.remove(best_checkpoint_path)