import json
import pdb
import numpy as np
import os

root = '../video_track'
ann_file = os.path.join(root, 'cctv_pipe_train.json')
with open(ann_file, 'r') as fp:
    data = json.load(fp)

test_keys = []
train_keys = []
for key in data['database'].keys():
    if data['database'][key]['subset'] == 'testing':
        test_keys.append(key)
    elif data['database'][key]['subset'] == 'training':
        train_keys.append(key)

np.random.shuffle(train_keys)
actual_train_keys = train_keys[:300]
val_keys = train_keys[300:]

with open('/tmp/train_keys_full.json', 'w') as fp:
    json.dump(actual_train_keys, fp)

with open('/tmp/val_keys_full.json', 'w') as fp:
    json.dump(val_keys, fp)

with open('/temp/test_keys.json', 'w') as fp:
    json.dump(test_keys, fp)
