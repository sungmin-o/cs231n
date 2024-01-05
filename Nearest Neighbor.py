import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class NearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    def predict(self, X):
        test_num = X.shape[0]
        Ypred = np.zeros(test_num, dtype = self.ytr.dtype)

        for i in range(test_num):
            print(i)
            distances = np.sum(np.abs(self.Xtr - X[i , :]), axis = 1)
            min_idx = np.argmin(distances)
            Ypred[i] = self.ytr[min_idx]
        return Ypred

Xtr = list()
ytr = list()

for i in range(1, 6):
    s = './cifar10/cifar-10-batches-py/data_batch_' + str(i)
    data_batch = unpickle(s)
    Xtr.extend(list(data_batch[b'data']))
    ytr.extend(list(data_batch[b'labels']))
Xtr = np.array(Xtr, dtype = np.float32)
ytr = np.array(ytr, dtype = np.int64)
test_batch = unpickle('./cifar10/cifar-10-batches-py/test_batch')
Xte = np.array(list(test_batch[b'data']), dtype = np.float32)
yte = np.array(list(test_batch[b'labels']), dtype = np.int64)

nn = NearestNeighbor()
nn.train(Xtr, ytr)

print(np.mean(nn.predict(Xte) == yte))