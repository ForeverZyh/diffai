import numpy as np
import itertools
import torch
import copy


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
                    else:
                        x2 = x.clone()
                    for swap_pos in swap_poss:
                        x2[swap_pos], x2[swap_pos + 1] = x2[swap_pos + 1], x2[swap_pos]
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