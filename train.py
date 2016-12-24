import argparse
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions
from document_reader import docs_to_index
import numpy as np
import pickle

class Encoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(Encoder, self).__init__(
                RNN=L.LSTM(n_units, n_units)
                )

    def reset():
        self.RNN.reset_state()

    def __call__(self, sentence):
        for word in sentence:
            context = self.RNN(word)
        return context

class Decoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(Decoder, self).__init__(
                RNN=L.LSTM(n_units, n_units),
                output_layer=L.Linear(n_units, n_vocab)
                )
        self.max_len=100

    def __call__(self, context, target, train):
        output = []
        loss = 0
        if train:
            for target_word in target:
                output_word = output_layer(context)
                t = chainer.Variable(np.array([target_word], dtype=np.int32))
                loss += F.softmax_cross_entropy(output_word, t)
                output += output_word.data
                return output, loss
        else:
            while next_word is not word2index(eos) and len(output) < self.max_len:
                output_word = output_layer(context)
                output += output_word.data
                return output

    def reset():
        self.RNN.reset_state()

class SkipThought(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        super(SkipThought, self).__init__(
                embed=L.EmbedID(n_vocab, n_units),
                encoder = Encoder(n_vocab, n_units),
                prev_decoder = Decoder(n_vocab, n_units),
                self_decoder = Decoder(n_vocab, n_units),
                next_decoder = Decoder(n_vocab, n_units)
        )
        self.train = train

    def __call__(self, input_sentences):
        print("++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-")
        print(input_sentences[:,1].data)
        context = self.encoder(self.embed(input_sentences[:,1]))
        loss = 0
        outputs = []
        if  self.train:
            for decoder, i in zip([prev_decoder, self_decoder, next_decoder], range(3)):
                o, l = decoder(context, input_sentences[:,1], input_sentences[:,i], self.train)
                loss += l
                outputs.append(o)
            return outputs, loss
        else:
            o, l = decoder(context, input_sentence, None, self.train)
            return o

class DocumentIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.data = np.asarray(list(self.gen_data(dataset)))
        self.itter = SerialIterator(self.data, batch_size, repeat, shuffle)

    def __next__(self):
        return self.itter.next()

    @property
    def epoch_detail(self):
        return self.itter.epoch_detail

    def gen_data(self, dataset):
        for doc in dataset:
            for i in range(1, len(doc)-1):
                yield (np.asarray(doc[i-1], dtype=np.int32),
                        np.asarray(doc[i], dtype=np.int32),
                        np.asarray(doc[i+1], dtype=np.int32))

class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
                train_iter, optimizer, device=device
        )
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range(self.bprop_len):
            batch = train_iter.__next__()
            x = self.converter(batch, self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(x))

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

f = open('vocab.bin', 'rb')
word2index = pickle.load(f)
f.close()
index2word = {wid: word for word, wid in word2index.items()}
n_vocab = len(word2index)

docs_data = docs_to_index(word2index, 'data')
iter = DocumentIterator(docs_data, 5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=1,
            help='batch size')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
            help='length of trancated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
            help='epoch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
            help='GPU ID')
    parser.add_argument('--gradclip', type=int, default=5,
            help='Gradient norm threshold to clip')
    parser.add_argument('--out', type=str, default='result',
            help='Directory to output the result')
    parser.add_argument('--source', '-s', type=str, default='data',
            help='path to data directory')
    parser.add_argument('--unit', '-u', type=int, default=650,
            help='Number of LSTM units in each layer')
    args = parser.parse_args()

    print("\n-------------------------------------------")

    skipthought = SkipThought(n_vocab, args.unit)
    model = L.Classifier(skipthought)

    docs_data = docs_to_index(word2index, args.source)
    train_iter = DocumentIterator(docs_data, args.batchsize)
    print("batchsize = {}".format(args.batchsize))

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy']
        ))
    trainer.extend(extensions.ProgressBar(
        update_interval=1
        ))

    trainer.run()


if __name__ == '__main__':
    main()