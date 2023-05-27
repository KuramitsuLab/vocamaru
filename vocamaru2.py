import os
import json
import re
import sys
import pandas as pd

from janome.tokenizer import Tokenizer

import copy
import argparse
from sentencepiece import sentencepiece_model_pb2 as model
from transformers import AutoTokenizer, T5Tokenizer

def dump_spm(tokenizer_path, save_path='local', dump_file='dump.jsonl'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.save_pretrained(save_path)
    m = model.ModelProto()
    m.ParseFromString(open(f"{save_path}/spiece.model", 'rb').read())
    with open(dump_file, 'w') as w:
        for id, piece in enumerate(m.pieces):
            d = {
                "id": id,
                "type": piece.type,
                "piece": piece.piece,
                "score": piece.score,            
            }
            print(json.dumps(d, ensure_ascii=False), file=w)

def test_spm(tokenizer_path, test_file='test.txt'):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    print(tokenizer_path, tokenizer)
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                tt = tokenizer.encode(line)
                print(len(tt), line, tt)

def rewrite_spm(tokenizer_path, save_path='local', dump_file='dump.jsonl', test_file='test.txt'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.save_pretrained(save_path)
    m = model.ModelProto()
    m.ParseFromString(open(f"{save_path}/spiece.model", 'rb').read())
    vocab_map = {}
    with open(dump_file) as f:
        for line in f.readlines():
            d = json.loads(line)
            vocab_map[d['id']] = d
    for id, piece in enumerate(m.pieces):
        d = vocab_map[id]
        piece.type = d['type']
        piece.piece = d['piece']
        piece.score = d['score']
    print(dir(m))
    print(list(m.ListFields()))
    with open(f"{save_path}/spiece.model", 'wb') as f:
        f.write(m.SerializeToString())
    tokenizer = T5Tokenizer(f"{save_path}/spiece.model",
                            extra_ids=0, additional_special_tokens=[])
    tokenizer.save_pretrained(save_path)
    test_spm(save_path, test_file=test_file)


def setup():
    parser = argparse.ArgumentParser(description='vocamaru')
#    parser.add_argument('files', type=str, nargs='+', help='files')
    parser.add_argument('--tokenizer_path', default='google/flan-t5-small')
    parser.add_argument('--save_path', default='local')
    parser.add_argument('--dump_file', default='dump.jsonl')
    parser.add_argument('--rewrite', default=None)
    parser.add_argument('--test_file', default='test.txt')
    hparams = parser.parse_args()  # hparams になる
    return hparams


def main():
    hparams = setup()
    #os.makedirs(hparams.save_path, exist_ok=True)
    if hparams.rewrite:
        rewrite_spm(hparams.tokenizer_path, 
                    hparams.save_path, hparams.rewrite, hparams.test_file)
    else:
        dump_spm(hparams.tokenizer_path, hparams.save_path, hparams.dump_file)


if __name__ == '__main__':
    main()
