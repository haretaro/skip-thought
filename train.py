import argparse
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions
from chainer import reporter, report
from document_reader import docs_to_index
import numpy as np
import pickle

xp = None

class Encoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True, n_layers=1, use_cudnn=True, dropout=0.5):
        super(Encoder, self).__init__(
                embed=L.EmbedID(n_vocab, n_units),
                rnn=L.NStepLSTM(n_layers, n_units, n_units, dropout, use_cudnn)#layer, in, out
                )
        self.train = train
        self.n_layers = n_layers
        self.n_units = n_units
        self.reset()

    def reset(self):
        self.hx = None
        self.cx = None

    def __call__(self, xs):
        batch_size = len(xs)
        if self.hx is None:
            self.hx = chainer.Variable(xp.zeros((self.n_layers, batch_size, self.n_units), dtype=np.float32))
        if self.cx is None:
            self.cx = chainer.Variable(xp.zeros((self.n_layers, batch_size, self.n_units), dtype=np.float32))
        xs_ = [self.embed(x) for x in xs]
        self.hx, self.cx, _ = self.rnn(self.hx, self.cx, xs_, self.train)
        return self.hx[0]

class Decoder(chainer.Chain):
    def __init__(self, n_vocab, n_units, stop_wid, train=True):
        super(Decoder, self).__init__(
                rnn=L.LSTM(n_units, n_units),
                output_layer=L.Linear(n_units, n_vocab)
                )
        self.max_len=100
        self.stop_wid = stop_wid

    def __call__(self, context, target=None, train=True):
        output = []
        loss = 0
        if train:
            length = max([len(x) for x in target])
            self.rnn.set_state(context, context)
            for i in range(length):
                context = self.rnn(context)
                output_word = self.output_layer(context)
                target_word = [target[j][i].data if i < len(target[j]) else self.stop_wid for j in range(len(target))]
                target_word = xp.asarray(target_word, dtype=np.int32)
                loss += F.softmax_cross_entropy(output_word, target_word)
            return output, loss
        else:
            output = []
            for i in range(self.max_len):
                context = self.rnn(context)
                output_word = [np.argmax(w_) for w_ in self.output_layer(context).data]
                output.append(output_word)
            return output

    def reset(self):
        self.rnn.reset_state()

class SkipThought(chainer.Chain):

    def __init__(self, n_vocab, n_units, stop_wid, train=True):
        super(SkipThought, self).__init__(
                embed=L.EmbedID(n_vocab, n_units),
                encoder = Encoder(n_vocab, n_units),
                prev_decoder = Decoder(n_vocab, n_units, stop_wid),
                self_decoder = Decoder(n_vocab, n_units, stop_wid),
                next_decoder = Decoder(n_vocab, n_units, stop_wid)
        )
        self.train = train
        self.loss = 0

    def __call__(self, input_sentences):

        context = self.encoder([s[1] for s in input_sentences])
        loss = 0
        outputs = []
        if  self.train:
            #TODO:range使わないで書けた気がする
            for decoder, i in zip([self.prev_decoder, self.self_decoder, self.next_decoder], range(3)):
                o, l = decoder(context, [s[i] for s in input_sentences], self.train)
                loss += l
                outputs.append(o)
            self.loss = loss
            report({'loss': self.loss}, self)
            return loss
        else:
            raise(NotImplemented)
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

    @property
    def epoch(self):
        return self.itter.epoch

    def gen_data(self, dataset):
        for doc in dataset:
            for i in range(1, len(doc)-1):
                yield (xp.asarray(doc[i-1], dtype=np.int32),
                        xp.asarray(doc[i], dtype=np.int32),
                        xp.asarray(doc[i+1], dtype=np.int32))

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
            x = [chainer.Variable(u) for u in batch]
            loss += optimizer.target(x)

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=2,
            help='batch size')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
            help='length of trancated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=100,
            help='epoch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
            help='GPU ID')
    parser.add_argument('--gradclip', type=int, default=5,
            help='Gradient norm threshold to clip')
    parser.add_argument('--out', type=str, default='result',
            help='Directory to output the result')
    parser.add_argument('--source', '-s', type=str, default='data',
            help='path to data directory')
    parser.add_argument('--unit', '-u', type=int, default=100,
            help='Number of LSTM units in each layer')
    parser.add_argument('--printreport', '-p', default=False, action='store_true')
    parser.add_argument('--resume', '-r', type=str, default='')
    args = parser.parse_args()

    global xp
    xp = cuda.cupy if args.gpu >= 0 else np

    f = open('vocab.bin', 'rb')
    word2index = pickle.load(f)
    f.close()
    index2word = {wid: word for word, wid in word2index.items()}
    n_vocab = len(word2index)

    docs_data = docs_to_index(word2index, 'data')

    print("\n-------------------------------------------")
    print('n_vocab = {}'.format(n_vocab))
    print('n_unit = {}'.format(args.unit))
    print('batch size = {}'.format(args.batchsize))
    print('stop_wid = {}'.format(word2index['\n']))


    model = SkipThought(n_vocab, args.unit, word2index['\n'])

    docs_data = docs_to_index(word2index, args.source)
    train_iter = DocumentIterator(docs_data, args.batchsize)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch')) #1epoch 毎にモデルを保存
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    if args.printreport is True:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss']
            ))
        trainer.extend(extensions.ProgressBar(
            update_interval=1
            ))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
