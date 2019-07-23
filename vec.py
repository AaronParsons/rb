#! /usr/bin/env python
import numpy as np
import struct

VEC_HEADER = '<iihh' # img count, img size, min, max

def read_vec(filename, width, height):
    f = open(filename)
    cnt,size,_,_ = struct.unpack(VEC_HEADER, f.read(12))
    f.read(1) # gap byte
    d = []
    for i in range(cnt):
        d.append(np.fromfile(f, dtype='<H', count=size))
        f.read(1) # gap byte
    d = np.array(d)
    d.shape = (cnt, width, height)
    return d

def save_vec(d, filename):
    assert(d.ndim >= 2)
    width,height = d.shape[-2:]
    size = width * height
    cnt = d.size / size
    header = struct.pack(VEC_HEADER, cnt, size, 0, 0)
    payload = '\x00'.join([d[i].astype('<H').tobytes() for i in range(cnt)])
    binary = '\x00'.join([header, payload])
    if type(filename) is str:
        filename = open(filename, 'w')
    filename.write(binary)

def merge_vec(infiles, outfile, width, height):
    data = [read_vec(f, width, height) for f in infiles]
    data = np.concatenate(data, axis=0)
    save_vec(data, outfile)
