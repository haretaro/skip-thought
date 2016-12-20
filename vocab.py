import os
import pickle

#dicにfilenameを読ませる
#知らない単語があったらdicに追加する
def read(dic, filename):
    with open(filename) as f:
        for line in f.readlines():
            for word in line.split(' '):
                if word not in dic.keys():
                    dic[word] = len(dic)
    return dic

#directoryにあるファイルを全て読む
def read_files(dic, directory):
    files = os.listdir(directory)
    for filename in files:
        filepath = os.path.join(directory, filename)
        read(dic, filepath)
        print(filepath)
    return dic

if __name__ == '__main__':
    out = 'vocab.bin' #辞書のファイル名
    word2index = read_files({}, 'data')
    index2word = {wid: word for word, wid in word2index.items()}
    with open(out,'wb') as f:
        pickle.dump(word2index, f)
