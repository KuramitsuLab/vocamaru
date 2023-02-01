# SentencePiece ボキャブラリ入れ替え

## 用意するもの

追加するボキャブラリファイル

__添付してあるファイル__

* `empty.txt`: 何も追加しない
* `nlp2023_py0500.txt`: Python語彙500
* `nlp2023_py2500.txt`: Python語彙2500
* `nlp2023_py5000.txt`: Python語彙5000

## ボキャブラリの消去

```
python3 vocamaru.py empty.txt --tokenizer_path=megagonlabs/t5-base-japanese-web-8k --save_path=meg8
```

__オプションの説明__
* `--tokenizer`: ベースのトークンナイザ
* `--save_path`: 保存先

## 新ボキャブラリの追加

```
python3 vocamaru.py nlp2023_py0500.txt --tokenizer_path=megagonlabs/t5-base-japanese-web-8k --skip_empty --save_path=meg8
```

__オプションの説明__
* `--tokenizer`: ベースのトークンナイザ
* `--save_path`: 保存先
* `--skip_empty`: 新ボキャブラリが足りないときはそのまま残す
* `--head_first`: ボキャブラリテーブルの先頭から置き換える
* `--enable_trim`: トリミングをする：字句の汎化を行う
* `--disable_number`: 数字の置き換えはしない
* `--disable_ja`: 日本語の重複語は置き換えない。（SentencePiece を信じる）
