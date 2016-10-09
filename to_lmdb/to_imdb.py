# -*- coding: utf-8 -*-
# author = sai
import lmdb
import cPickle
from audio_pro import audio_frame_pb2
import numpy as np

audio_dir = './audio_pkl/audio.pkl'

def unpickle(file):
    with open(file, 'rb') as f:
        data, labels = cPickle.load(f)
    return  data, labels.astype(np.float)

def array2audio(arr, label=None):
    audio = audio_frame_pb2.Audio()
    audio.length, audio.channels = arr.shape
    if arr.dtype==np.uint8:
        audio.data = arr.tostring()
    else:
        arr = arr.astype(np.float)
        audio.float_data.extend(arr.flat)
    if audio.label is not None:
        label = np.argmax(label)
        audio.label = label
    return audio

if __name__=='__main__':
    import os, logging
    logging.basicConfig(level=logging.INFO)

    train_lmdb = 'audio_lmdb'
    if not os.path.exists(train_lmdb):
        os.mkdir(train_lmdb)
        logging.info('mkdir {}'.format(train_lmdb))

    data, labels = unpickle(audio_dir)
    logging.info('load data complete!')

    # operate lmdb
    env = lmdb.open(train_lmdb, 1024*1024*1024)
    txn = env.begin(write=True)
    count = 0

    for i in xrange(data.shape[0]):
        audio = array2audio(data[i], labels[i])
        str_id = '{:08}'.format(count)
        txn.put(str_id, audio.SerializeToString())
        count += 1

        if count%1000 == 0:
            logging.info('already handled with {} frames'.format(count))
            txn.commit()
            txn=env.begin(write=True)
    txn.commit()
    env.close()



