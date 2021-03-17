import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc
import json
from torch_geometric.datasets import TUDataset
import os.path as osp

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG')
idxs = json.load(open('test.json', 'r'))
dataset = dataset[idxs]
#dataset = build_dataset_from_name('mutag')
#idxs = json.load(open('test.json'))
train_split = len(dataset) // 10 * 8
dataset.train_index = list(range(train_split))
dataset.val_index = list(range(train_split, train_split + len(dataset) // 10))
dataset.test_index = list(range(len(dataset) - len(dataset) // 10, len(dataset)))
dataset.train_split = dataset[dataset.train_index]
dataset.val_split = dataset[dataset.val_index]
dataset.test_split = dataset[dataset.test_index]

from tqdm import tqdm
import numpy as np

acc_list = []

with tqdm(range(100)) as t:
    for _ in t:
        autoClassifier = AutoGraphClassifier.from_config('../configs/gin_default.yaml')

        # train
        autoClassifier.fit(
            dataset, 
            time_limit=3600, 
            cross_validation=False,
            evaluation_method=['acc']
        )

        # test
        predict_result = autoClassifier.predict_proba(use_best=True, use_ensemble=False)
        acc = Acc.evaluate(predict_result, dataset.data.y[dataset.test_index].cpu().detach().numpy())
        acc_list.append(acc)
        t.set_postfix(now=acc, mean=sum(acc_list) / len(acc_list), std=np.std(acc_list) if len(acc_list) > 1 else 0)
print(np.mean(acc_list), np.std(acc_list))
