import argparse
import chainer
from chainer import cuda
from chainer import training
from document_reader import docs_to_index, doc_to_index
import numpy as np
import pickle
import train
from train import SkipThought, BPTTUpdater, DocumentIterator

train.xp = np
xp = np

f = open('vocab.bin', 'rb')
word2index = pickle.load(f)
f.close()
index2word = {wid: word for word, wid in word2index.items()}
n_vocab = len(word2index)

def load_model(snap_shot, n_vocab, unit=100, batchsize=2, bproplen=35, gpu=-1, epoch=100, out='result', data_directory='data'):
    docs_data = docs_to_index(word2index, data_directory)
    model = SkipThought(n_vocab, unit, word2index['\n'])
    train_iter = DocumentIterator(docs_data, batchsize)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = BPTTUpdater(train_iter, optimizer, bproplen, gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    chainer.serializers.load_npz(snap_shot, trainer)
    return model

def to_vector(model, sentence):
    model.encoder.reset()
    vec = model.encoder(xp.asarray([sentence], dtype=np.int32))
    return vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
    parser.add_argument('--unit', '-u', type=int, default=100)
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--batchsize', '-b', type=int, default=2)
    parser.add_argument('--bproplen', '-l', type=int, default=35, help='length of trancated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--out', type=str, default='result', help='Directory to output the result')
    args = parser.parse_args()

    model = load_model(args.resume, n_vocab, args.unit, args.batchsize, args.bproplen, args.gpu, args.epoch, args.out, 'data')

    doc = doc_to_index(word2index, 'data/wagahaiwa_nekodearu.txt')
    doc = doc_to_index(word2index, 'data/kappa.txt')

    vec = to_vector(model, doc[30])
    print([index2word[x] for x in doc[30]])

    out = model.prev_decoder(vec, train=False)
    out = [index2word[x[0]] for x in out]
    print(out)

    out = model.self_decoder(vec, train=False)
    out = [index2word[x[0]] for x in out]
    print(out)

    out = model.next_decoder(vec, train=False)
    out = [index2word[x[0]] for x in out]
    print(out)
