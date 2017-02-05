import argparse
import chainer
from chainer import cuda
from chainer import training
from document_reader import docs_to_index
import numpy as np
import pickle
import train
from train import SkipThought, BPTTUpdater, DocumentIterator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--unit', '-u', type=int, default=100)
parser.add_argument('--resume', '-r', type=str, default='')
parser.add_argument('--batchsize', '-b', type=int, default=2)
parser.add_argument('--bproplen', '-l', type=int, default=35, help='length of trancated BPTT')
parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
parser.add_argument('--out', type=str, default='result', help='Directory to output the result')
args = parser.parse_args()

xp = cuda.cupy if args.gpu >= 0 else np
train.xp = xp
f = open('vocab.bin', 'rb')
word2index = pickle.load(f)
f.close()
index2word = {wid: word for word, wid in word2index.items()}
n_vocab = len(word2index)
model = SkipThought(n_vocab, args.unit, word2index['\n'])
docs_data = docs_to_index(word2index, 'data')
train_iter = DocumentIterator(docs_data, args.batchsize)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
chainer.serializers.load_npz(args.resume, trainer)

in_data = [word2index[x] for x in ['吾輩', 'は', '猫', 'で', 'ある', '\n']]
in_data = [word2index[x] for x in ['名前', 'は', 'まだ', 'ない', '\n']]
print(in_data)
model.encoder.reset()
vec = model.encoder(xp.asarray([in_data], dtype=np.int32))
print(vec.data)

model.self_decoder.reset()
out = model.self_decoder(vec, train=False)
print(out)

out_words = [index2word[x[0]] for x in out]
print(out_words)


model.next_decoder.reset()
out = model.next_decoder(vec, train=False)
out_words = [index2word[x[0]] for x in out]
print(out_words)
