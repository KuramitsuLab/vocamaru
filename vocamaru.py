import os
import json
import re
import sys
import pandas as pd

from janome.tokenizer import Tokenizer

import copy
import argparse
from sentencepiece import sentencepiece_model_pb2 as model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer

LOG = None


def println(*args, **kwargs):
    print(*args, **kwargs)
    if LOG:
        print(*args, file=LOG)


def log(*args, **kwargs):
    print('[LOG]', *args, **kwargs)
    if LOG:
        print('[LOG]', *args, file=LOG)


def read_new_vocab(files, vocab_map):
    new_vocab = {}
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                if '\t' in line or ' ' in line:
                    line = line.split()[0]
                else:
                    line = line.strip()
                if line == '' or line in vocab_map:
                    continue
                if line not in new_vocab:
                    new_vocab[line] = line
                else:
                    log('登録済み', line)
    println('[置き換える語彙数]', len(new_vocab))
    return list(new_vocab.keys())


t = Tokenizer()

pDigit = re.compile('^[0-9]+$')
pAlpha = re.compile('^[A-Za-z]+$')
pHira = re.compile('[ぁ-ん]')


def containsHira(w):
    return bool(re.search(pHira, w))


def transform(w, s):
    if re.search(pDigit, w):
        return w, '数字'
    if re.search(pAlpha, w):
        return w, '英字'
    return w, s


def janome2(s):
    ws = []
    ss = []
    if s.startswith('▁'):
        s = s[1:]
    for token in t.tokenize(s):
        # 字句と品詞で別のリストを返す
        w, s = transform(token.surface, token.part_of_speech.split(',')[0])
        ws.append(w)
        ss.append(s)
    return ws, ' '.join(ss)


IKI = set(['連体詞 助詞', '助詞 助詞', '動詞 助動詞', '動詞 助動詞 助動詞', '動詞 動詞', '助動詞 助動詞', '動詞 動詞 助動詞', '名詞 助動詞', '名詞 助動詞 助動詞',
          '名詞 名詞', '名詞 名詞 名詞', '名詞 名詞 名詞 名詞', '名詞', '動詞', '副詞', '形容詞', '助詞', '接続詞', '連体詞', '助動詞', '感動詞', 'フィラー', '接頭詞'])


def get_trim(ws, vocab_map):
    for w in ws:
        if w not in vocab_map:
            return w
    return None


REMOVE_JAPANESE = True
REMOVE_SYMBOL = True
REMOVE_NUMBERS = True
ENABLE_TRIM = False


def remove_vocab(vocab_map, removed_map, new_set):
    trimed = {}
    # 論文アルゴリズムによる重複語の削除
    before = len(removed_map)
    if REMOVE_JAPANESE:
        # トリムが何だったのか思い出せない？
        for token in vocab_map.keys():
            if token in new_set:
                continue
            if containsHira(token):
                ws, pos = janome2(token)
                idx = vocab_map[token]
                if pos not in IKI:
                    trim = get_trim(ws, vocab_map)
                    if trim and trim not in trimed:
                        # removed_map[token] = trim
                        if ENABLE_TRIM:
                            trimed[trim] = token
                            log(f'トリム {token} => {trim}')
                    else:
                        removed_map[token] = f'<empty_{idx}>'
    println('[重複語数]', len(removed_map)-before, 'トリム数', len(trimed))

    def remove_gomi(chars, prespace=True):
        nonlocal vocab_map, removed_map, new_set
        cc = 0
        for token, idx in vocab_map.items():
            if token in removed_map or token in new_set:
                continue
            for c in chars:
                if c in token:
                    if c == token or c in new_set or (prespace and token == f'▁{c}'):
                        continue
                    if token not in removed_map:
                        removed_map[token] = f'<empty_{idx}>'
                        cc += 1
        return cc
    # 記号ゴミの削除
    before = len(removed_map)
    if REMOVE_SYMBOL:
        println('全角ゴミ', remove_gomi("。、．・〜（）”“【】「」『』［］｛｝♪～〖〗"))
        println('半角ゴミ', remove_gomi("'+-()[]{}!#$%&=~|`;:,.@?<>\'\"\\'*/_"))
        println('記号ゴミ', remove_gomi("°±¤©＋–×÷£€¢¬●′‚·¶«"))
    println('[記号ゴミ]', len(removed_map)-before)
    # 数字ゴミ
    before = len(removed_map)
    if REMOVE_NUMBERS:
        println('数字ゴミ', remove_gomi("0123456789"))
    println('[数字重複]', len(removed_map)-before)
    return trimed


def append_extra_ids(m):
    found_extra_ids = False
    for id, piece in enumerate(m.pieces):
        token = piece.piece
        if '<extra_id_' in token:
            found_extra_ids = True
            if token.startswith('▁'):
                piece.piece = piece.piece[1:]
            print(token, id, piece.type, piece.score)
            return
    if found_extra_ids:
        return
    for id in range(99, -1, -1):
        p = copy.copy(m.pieces[0])
        p.piece = f'<extra_id_{id}>'
        p.type = 4
        p.score = 0.0
        m.pieces.append(p)
    # print(len(m.pieces), type(m.pieces), dir(m.pieces))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


TAIL_FIRST = True
SKIP_EMPTY = False

# https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/#download-the-pretrained-model


def replace_vocab(files, tokenizer_path, save_path='local'):
    # トークンナイザーのコピーをsave_pathに作る
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_path)
    println('[パラメータ数]', tokenizer_path, count_parameters(pretrained))
    pretrained.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    # tokenizer.special_tokens_map_file = "special_tokens_map.json"
    if 'special_tokens_map_file' in tokenizer.init_kwargs:
        tokenizer.init_kwargs['special_tokens_map_file'] = 'special_tokens_map.json'
    if 'additional_special_tokens' in tokenizer.init_kwargs:
        # tokenizer.init_kwargs['additional_special_tokens']=[]
        del tokenizer.init_kwargs['additional_special_tokens']
    if 'extra_ids' in tokenizer.init_kwargs:
        del tokenizer.init_kwargs['extra_ids']

    print(tokenizer.init_kwargs)

    tokenizer.additional_special_tokens = []
    tokenizer.additional_special_tokens_ids = []
    tokenizer.save_pretrained(save_path)
    println('[新しいモデルの保存先]', save_path)

    # 語彙テーブルを開けて情報を取り出す
    m = model.ModelProto()
    m.ParseFromString(open(f"{save_path}/spiece.model", 'rb').read())
    vocab_map = {}  # 字句とIDの辞書
    ss = []
    for id, piece in enumerate(m.pieces):
        if piece.type == 1:
            token = piece.piece
            vocab_map[token] = id
            ss.append(piece.score)
        elif piece.type != 6:
            log(f'特殊語彙 id={id} type={piece.type} {piece.piece} {piece.score}')
    println('[全語彙数]', len(vocab_map))
    df = pd.DataFrame({'s': ss})
    println('[スコア統計]', df.describe())

    # 追加する字句を読む
    new_vocab = read_new_vocab(files, vocab_map)

    removed_map = {}
    trimed = remove_vocab(vocab_map, removed_map, set(new_vocab))
    println('[消去可能な字句]', len(removed_map))

    ss = []
    for id, piece in enumerate(m.pieces):
        if piece.type == 1:
            token = piece.piece
            if token in removed_map:
                ss.append((piece.score, token))
    ss.sort()
    removed_tokens = [t for _, t in ss]

    if TAIL_FIRST:
        new_tokens = [t for t in new_vocab if t not in trimed][::-1]
    else:
        removed_tokens = removed_tokens[::-1]
        new_tokens = [t for t in new_vocab if t not in trimed][::-1]
    println('[実際に置き換える語]', len(new_tokens))

    with open(f'{save_path}/removed.jsonl', 'w') as w:
        for newtoken, oldtoken in trimed.items():
            idx = vocab_map[oldtoken]
            m.pieces[idx].piece = newtoken
            d = {'in': newtoken, 'out': oldtoken,
                 'idx': idx, 'score': m.pieces[idx].score}
            println(json.dumps(d, ensure_ascii=False), file=w)
        for token in removed_tokens:
            idx = vocab_map[token]
            if len(new_tokens) != 0:
                newtoken = new_tokens.pop()
            else:
                if SKIP_EMPTY:
                    break
                newtoken = removed_map[token]
            m.pieces[idx].piece = newtoken
            d = {'in': newtoken, 'out': token,
                 'idx': idx, 'score': m.pieces[idx].score}
            println(json.dumps(d, ensure_ascii=False), file=w)
    append_extra_ids(m)
    # mT5用のおまじない
    # if len(m.pieces) > 250000:
    #     for i, piece in enumerate(m.pieces[250000:], 250000):
    #         if 'extra_id' in piece.piece:
    #             piece.piece = piece.piece[1:]
    with open(f"{save_path}/spiece.model", 'wb') as f:
        f.write(m.SerializeToString())
    tokenizer = T5Tokenizer(f"{save_path}/spiece.model",
                            extra_ids=0, additional_special_tokens=[])
    tokenizer.save_pretrained(save_path)
    test_vocab(save_path, new_vocab)


def test_vocab(tokenizer_path, new_vocab):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    println(tokenizer_path, tokenizer)
    for v in new_vocab:
        tt = tokenizer.encode(v)
        if len(tt) > 3:
            println('[ミス]', v, len(tt), tt)
    for v in ['<nl><nl>', '<123> <100> <1>', '<extra_id_0><extra_id_99>', '\n\t\n']:
        println(v, tokenizer.encode(v))


def setup():
    global REMOVE_JAPANESE, REMOVE_SYMBOL, REMOVE_NUMBERS, TAIL_FIRST, SKIP_EMPTY, ENABLE_TRIM
    parser = argparse.ArgumentParser(description='vocamaru')
    parser.add_argument('files', type=str, nargs='+', help='files')
    parser.add_argument('--tokenizer_path', default='google/mt5-small')
    parser.add_argument('--save_path', default='local')
    parser.add_argument('--disable_ja', action='store_true', default=False)
    parser.add_argument('--disable_symbol', action='store_true', default=False)
    parser.add_argument('--disable_number', action='store_true', default=False)
    parser.add_argument('--head_first', action='store_true', default=False)
    parser.add_argument('--skip_empty', action='store_true', default=False)
    parser.add_argument('--enable_trim', action='store_true', default=False)
    hparams = parser.parse_args()  # hparams になる
    REMOVE_JAPANESE = not hparams.disable_ja
    REMOVE_SYMBOL = not hparams.disable_symbol
    REMOVE_NUMBERS = not hparams.disable_number
    TAIL_FIRST = not hparams.head_first
    SKIP_EMPTY = hparams.skip_empty
    ENABLE_TRIM = hparams.enable_trim
    return hparams


def main():
    global LOG
    hparams = setup()
    log(hparams)
    os.makedirs(hparams.save_path, exist_ok=True)
    with open(f'{hparams.save_path}/vocamaru_log.txt', 'w') as f:
        LOG = f
        replace_vocab(hparams.files, hparams.tokenizer_path, hparams.save_path)
        LOG = sys.stdout


if __name__ == '__main__':
    main()
