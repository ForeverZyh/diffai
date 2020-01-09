import numpy as np
import itertools
import torch

from utils import swap_pytorch
from dataset.dataset_loader import SSTWordLevel, Glove


dict_map = dict(np.load("./dataset/AG/dict_map.npy").item())
lines = open("./dataset/en.key").readlines()
adjacent_keys = [[] for i in range(len(dict_map))]
for line in lines:
    tmp = line.strip().split()
    ret = set(tmp[1:]).intersection(dict_map.keys())
    ids = []
    for x in ret:
        ids.append(dict_map[x])
    adjacent_keys[dict_map[tmp[0]]].extend(ids)
    
def SwapSub(a, b, x, is_numpy=False, batch_size=64):
    if not is_numpy:
        x = x.cpu()
        X = []
    else:
        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
        current_id = 0
    valid_swap_poss = [i for i in range(len(x) - 1) if int(x[i]) != int(x[i + 1])]
    for swap in range(a, -1, -1):
        for swap_poss in itertools.combinations(tuple(valid_swap_poss), swap):
            # precheck whether overlape
            overlape = False
            for i in range(len(swap_poss) - 1):
                if swap_poss[i + 1] - swap_poss[i] == 1:
                    overlape = True
            if overlape:
                continue
            valid_sub_poss = [i for i in range(len(x)) if (i not in swap_poss) and (i - 1 not in swap_poss) and len(adjacent_keys[int(x[i])]) > 0]
            for sub in range(b, -1, -1):
                for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
                    if is_numpy:
                        x2 = X[current_id]
                        for swap_pos in swap_poss:
                            x2[swap_pos], x2[swap_pos + 1] = x2[swap_pos + 1], x2[swap_pos]
                    else:
                        x2 = x.clone()
                        for swap_pos in swap_poss:
                            swap_pytorch(x2, swap_pos, swap_pos + 1)
                    for sub_pos in sub_poss:
                        x2[sub_pos] = adjacent_keys[int(x[sub_pos])][0]
                    if is_numpy:
                        current_id += 1
                        if current_id >= batch_size:
                            yield X
                            X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
                            current_id = 0
                    else:
                        X.append(x2.unsqueeze(0))
                        if len(X) == batch_size:
                            yield torch.cat(X, 0).cuda()
                            X = []
    if len(X) > 0:
        if is_numpy:
            yield X
        else:
            yield torch.cat(X, 0).cuda()

            
def DelDupSubWord(a, b, c, x, is_numpy=False, batch_size=64, del_set={"a", "and", "the", "of", "to"}, padding_id=0):
    SSTWordLevel.build()
    if not is_numpy:
        x = x.cpu()
        X = []
    else:
        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
        current_id = 0
    end_pos = len(x)
    while end_pos > 0 and int(x[end_pos - 1]) == padding_id:
        end_pos -= 1
        
    valid_sub_poss = [i for i in range(end_pos) if Glove.id2str[int(x[i])] in SSTWordLevel.synonym_dict]
    for sub in range(c, -1, -1):
        for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
            if is_numpy:
                x3 = x.copy()
            else:
                x3 = x.clone()
            for sub_pos in sub_poss:
                x3[sub_pos] = Glove.str2id[SSTWordLevel.synonym_dict[Glove.id2str[int(x[sub_pos])]][0]]
            valid_dup_poss = [i for i in range(end_pos) if i not in sub_poss]
            for dup in range(b, -1, -1):
                for dup_poss in itertools.combinations(tuple(valid_dup_poss), dup):
                    valid_del_poss = [i for i in range(end_pos) if (i not in dup_poss) and (i not in sub_poss) and Glove.id2str[int(x[i])] in del_set]
                    for delete in range(a, -1, -1):
                        for del_poss in itertools.combinations(tuple(valid_del_poss), delete):
                            if is_numpy:
                                x2 = X[current_id]
                            else:
                                x2 = x.clone()
                            copy_point = 0
                            paste_point = 0
                            while copy_point < end_pos and paste_point < end_pos:
                                if copy_point in dup_poss:
                                    x2[paste_point] = x3[copy_point]
                                    paste_point += 1
                                    if paste_point < end_pos:
                                        x2[paste_point] = x3[copy_point]
                                        paste_point += 1
                                        copy_point += 1
                                elif copy_point in del_poss:
                                    copy_point += 1
                                else:
                                    x2[paste_point] = x3[copy_point]
                                    paste_point += 1
                                    copy_point += 1
                                    
                            while paste_point < end_pos:
                                x2[paste_point] = padding_id
                                paste_point += 1
                                    
                            if is_numpy:
                                current_id += 1
                                if current_id >= batch_size:
                                    yield X
                                    X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
                                    current_id = 0
                            else:
                                X.append(x2.unsqueeze(0))
                                if len(X) == batch_size:
                                    yield torch.cat(X, 0).cuda()
                                    X = []

    if len(X) > 0:
        if is_numpy:
            yield X
        else:
            yield torch.cat(X, 0).cuda()
