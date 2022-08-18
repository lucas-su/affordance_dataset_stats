from scipy.io import loadmat
import pickle
import json
import numpy as np
from multiprocessing import Pool
from collections import Counter

def getmat(path):
    p_l = 0
    p_u = 0
    f = loadmat(f'{path}/seg.mat')
    for i, obj in enumerate(f['names'][0]):
        if obj in labels['labels']:
            p_l += np.count_nonzero(f['seglabel']==i+1)
        else:
            p_u += np.count_nonzero(f['seglabel']==i+1)
    return p_l, p_u


if __name__ == '__main__':
    p_labeled = 0
    p_unlabeled = 0
    root = "/media/luc/data/sunrgbd"
    with open("./annotated_labels.json") as file:
        labels = json.load(file)
    with open(root + '/trainfile_dirs.pickle', 'rb') as file:
        train_fp = pickle.load(file)
    with open(root + '/valfile_dirs.pickle', 'rb') as file:
        val_fp = pickle.load(file)
    with open(root + '/testfile_dirs.pickle', 'rb') as file:
        test_fp = pickle.load(file)
    filepaths = train_fp.copy()
    filepaths.extend(val_fp)
    filepaths.extend(test_fp)
    #
    # with Pool(5) as p:
    #     results = p.map(getmat, filepaths)
    # for res in results:
    #     p_labeled += res[0]
    #     p_unlabeled += res[1]
    # print(p_labeled,p_unlabeled, "\n\n")


    count_u = Counter()
    count_l = Counter()
    for path in set(filepaths):
        cnt_u_file = Counter()
        cnt_l_file = Counter()
        f = loadmat(f'{path}/seg.mat')
        for i, obj in enumerate(f['names'][0]):
            if obj not in labels['labels']:
                cnt_u_file.update(Counter({f"{obj}":np.count_nonzero(f['seglabel']==i+1)}))
            else:
                cnt_l_file.update(Counter({f"{obj}":np.count_nonzero(f['seglabel']==i+1)}))
        count_u.update(cnt_u_file)
        count_l.update(cnt_l_file)
    print(count_u.__len__())
    print(count_l.__len__())
