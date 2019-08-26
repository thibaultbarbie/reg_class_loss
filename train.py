import chainer
from chainer import iterators, optimizers, report, training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
from chainer.datasets import mnist


class NeuralNetwork(chainer.Chain):
    def __init__(self):
        super(NeuralNetwork, self).__init__(
            fc=L.Linear(None, 100),
            regression=L.Linear(None, 784),
            classification = L.Linear(None, 10)
        )
        
    def forward(self, x):
        h = F.relu(self.fc(x))
        regression_result = self.regression(h)
        class_result = self.classification(h)
        return regression_result, class_result
        
    def __call__(self, x, label):
        reg_result, class_result = self.forward(x)
        loss_regression = F.mean_squared_error(reg_result, x)
        loss_classification = F.softmax_cross_entropy(class_result, label)
        loss_total = loss_regression + loss_classification
        
        report({'loss_regression': loss_regression}, self)
        report({'loss_classification': loss_classification}, self)
        report({'loss_total': loss_total}, self)
        return loss_total


def train():
    max_epoch = 10
    batchsize = 1000
    
    model = NeuralNetwork()
    
    train, test = mnist.get_mnist()
    train_iter = iterators.SerialIterator(train, batchsize)
    
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    updater = training.updaters.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'))
    
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss_total', 'main/loss_regression', 'main/loss_classification', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__=="__main__":
    train()
