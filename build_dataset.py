from scipy.io import loadmat
import pickle
import json
import numpy as np
import pandas as pd
import csv
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
    transfer = {} # dict going from object label to aff label
    root = "/media/luc/data/sunrgbd"
    with open("./annotated_labels.json") as file:
        labels = json.load(file)
        labels = labels['labels'] # make list
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

    with open('/media/luc/data/sunrgbd/colnames.pylist', 'r') as file:
        cols = eval(file.read())
    with open('/media/luc/data/sunrgbd/web_annotations.csv', 'r') as file:
        annotations = pd.DataFrame(list(csv.reader(file)), columns=cols)


    affordances = ["con_move", "uncon_move", "dir_affs", "indir_affs", "observe_affs", "social_affs", "no_affs",
                   "no_clue", "roll", "push", "drag", "tether", "pick_up_carry", "pour", "fragile", "open", "grasp",
                   "pull", "tip", "stack", "cut_scoop", "support", "transfer", "requires_other", "info", "deco",
                   "together", "none", "warmth", "illumination", "walk"]

    GT = pd.DataFrame(columns=['id', 'object'] + affordances)

    for row in annotations.iterrows():
        if row[1]['anno_1_id'] != "NULL":
            aff_c = Counter()
            for affordance in affordances:
                for ann in [1,2,3]: # there are (at most) three annotators per object label, starting count at 1
                    if row[1][f'anno_{ann}_{affordance}'] == '1':
                        aff_c.update([affordance])
                    elif row[1][f'anno_{ann}_{affordance}'] == '0':
                        aff_c.subtract([affordance])
            GT.loc[row[0], 'id'] = row[1]['id']
            GT.loc[row[0],'object'] = row[1]['object_label']
            for element in aff_c:
                GT.loc[row[0],element]= next(x if not x<0 else 0 for x in [aff_c[element]])


    for path in set(filepaths):
        f = loadmat(f'{path}/seg.mat')



    #     for i, obj in enumerate(f['names'][0]):
    #         if obj not in labels['labels']:
    #             cnt_u_file.update(Counter({f"{obj}":np.count_nonzero(f['seglabel']==i+1)}))
    #         else:
    #             cnt_l_file.update(Counter({f"{obj}":np.count_nonzero(f['seglabel']==i+1)}))
    #     count_u.update(cnt_u_file)
    #     count_l.update(cnt_l_file)
    # print(count_u.__len__())
    print('done')