import collections
import pickle
import pandas as pd


def change_word_to_id(input_word: str, word_id_dict: dict) -> str:
    # ID番号へ変換、辞書に存在しないものは0をいれる
    result_list = []
    for word in input_word.split():
        if word in word_id_dict:
            result_list.append(str(word_id_dict[word]))
        else:
            result_list.append("0")

    return " ".join(result_list)


# データの読み込み
train = pd.read_csv("train.txt", sep="\t", quoting=False,
                    names=["TITLE", "CATEGORY"])
test = pd.read_csv("test.txt", sep="\t", quoting=False,
                   names=["TITLE", "CATEGORY"])

# trainとtestを結合する
train["flg"] = "train"
test["flg"] = "test"
train_test = pd.concat([train, test])


# 全文章を一つにまとめたstrを生成
all_sentence_list = " ".join(train_test["TITLE"].tolist()).split(" ")

# 全文章に含まれる単語の頻度を計算
all_word_cnt = collections.Counter(all_sentence_list)

# 出現頻度が2回以上の単語のみを取得
word_cnt_over2 = [i for i in all_word_cnt.items() if i[1] >= 2]
word_cnt_over2 = sorted(word_cnt_over2, key=lambda x: x[1], reverse=True)

# 単語のみ取得
word_over2 = [i[0] for i in word_cnt_over2]
# ID番号を取得
id_list = [i for i in range(1, len(word_over2))]

# 単語とID番号をdictへとまとめる
word_id_dict = dict(zip(word_over2, id_list))

# 出力
with open("word_id_dict.pkl", "wb") as tf:
    pickle.dump(word_id_dict, tf)

# train_testのTITLEをID番号へと変換
train_test["TITLE"] = train_test["TITLE"].apply(
    lambda x: change_word_to_id(x, word_id_dict)
)

# train, testへ分離
train = train_test.query('flg == "train"')
test = train_test.query('flg == "test"')

# 出力
train.to_pickle("train_title_id.pkl")
train.to_csv("train_title_id.csv")

test.to_pickle("test_title_id.pkl")
test.to_csv("test_title_id.csv")
