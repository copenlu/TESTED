import os
import os.path as osp
import json
import time
import enum
from collections import defaultdict
import subprocess


from typing import List, Tuple

from collections import OrderedDict
import xml.etree.ElementTree
import numpy as np
import torch

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_segmentor(seq, size):
    newseq = []
    splitsize = 1.0/max(1,size)*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res


def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

def check_gpu():
    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    return ('\nUpdated at %s\nGPU utilization: %s %%\nVRAM used: %s %%\n%s\n' % (now, d["gpu_util"],d["mem_used_per"], msg))


def write_jsonl(data: List = [], data_path:str = '',  data_key:str = '') -> int:
    success = False
    try:
        with open(f'{data_path}/{data_key}.jsonl', 'w') as outfile:
            for line in data:
                json.dump(line, outfile)
                outfile.write('\n')

        success = True
    except RuntimeError as e:
        print('Cannot Write a jsonL for data with error: ', e)
        return success
    return success


def insert_scheme(str1: str, str2: str, scheme: str) -> str:
    return scheme.replace('<premise>', str1).replace('<hypothesis>', str2)


