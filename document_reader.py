import os
import pickle

#小説をword id の列に変換する
def doc_to_index(word2index, filename):
    seq = []
    with open(filename) as f:
        for line in f.readlines():
            l = []
            for word in line.split(' '):
                l.append(word2index[word])
            seq.append(l)
    return seq

#directory にあるファイルを全てword id の列に変換
def docs_to_index(word2index, directory):
    seq = []
    files = os.listdir(directory)
    for filename in files:
        targetpath = os.path.join(directory, filename)
        seq.append(doc_to_index(word2index, targetpath))
    return seq


if __name__ == '__main__':
    f = open('vocab.bin', 'rb')
    word2index = pickle.load(f)
    f.close()
    index2word = {wid: word for word, wid in word2index.items()}

    seq = docs_to_index(word2index, 'data')

    for doc in seq:
        for line in doc:
            words = ""
            for wid in line:
                words += index2word[wid]
            print(words)

