import numpy as np
np.random.seed(133)
class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.

    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples,)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]

class BatchGeneratorFeatureTemp(object):
    def __init__(self, X,temp, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(temp) != np.ndarray:
            temp = np.asarray(temp)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._temp =temp
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._temp =self._temp[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def temp(self):
        return self._temp
    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples,)
                self._X = self._X[new_index]
                self._temp = self._temp[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._temp[start:end],self._y[start:end]

class BatchGeneratorTransferLearning(object):
    def __init__(self, X,lineNum,temp_before,temp_after, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(lineNum)!= np.ndarray:
            lineNum =np.ndarray(lineNum)
        if type(temp_before) != np.ndarray:
            temp_before =np.ndarray(temp_before)
        if type(temp_after) != np.ndarray:
            temp_after = np.asarray(temp_after)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._lineNum =lineNum
        self._temp_before =temp_before
        self._temp_after =temp_after
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._lineNum =self._lineNum[new_index]
            self._temp_before =self._temp_before[new_index]
            self._temp_after =self._temp_after[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X
    @property
    def lineNum(self):
        return self._lineNum
    @property
    def temp_before(self):
        return self._temp_before
    @property
    def temp_after(self):
        return self._temp_after
    @property
    def y(self):
        return self._y
    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples,)
                self._X = self._X[new_index]
                self._lineNum =self._lineNum[new_index]
                self._temp_before = self._temp_before[new_index]
                self._temp_after = self._temp_after[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._lineNum[start:end],self._temp_before[start:end],self._temp_after[start:end],self._y[start:end]

class BatchGeneratorTransferLearningTemp(object):
    def __init__(self,lineNum,temp_before,temp_after, y, shuffle=False):
        if type(lineNum)!= np.ndarray:
            lineNum =np.ndarray(lineNum)
        if type(temp_before) != np.ndarray:
            temp_before =np.ndarray(temp_before)
        if type(temp_after) != np.ndarray:
            temp_after = np.asarray(temp_after)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._lineNum =lineNum
        self._temp_before =temp_before
        self._temp_after =temp_after
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._y.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._lineNum =self._lineNum[new_index]
            self._temp_before =self._temp_before[new_index]
            self._temp_after =self._temp_after[new_index]
            self._y = self._y[new_index]

    @property
    def lineNum(self):
        return self._lineNum
    @property
    def temp_before(self):
        return self._temp_before
    @property
    def temp_after(self):
        return self._temp_after
    @property
    def y(self):
        return self._y
    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples,)
                self._lineNum =self._lineNum[new_index]
                self._temp_before = self._temp_before[new_index]
                self._temp_after = self._temp_after[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._lineNum[start:end],self._temp_before[start:end],self._temp_after[start:end],self._y[start:end]

