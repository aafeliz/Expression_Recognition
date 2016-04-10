import numpy as np
import math
import pandas

'''
    def fit(self, X, y, sig_, epochs, nu):  # x comes in as 1-D array of 2399

        amount =np.bincount(y[:, 0])
        for i in np.unique(y):
            indx = 0
            xinput = np.zeros(shape=(amount[i]))
            for j in range(X.shape[0]):  #2399
                class_ = y[j, 0]         # class for the sample number j
                if class_ == i:
                    xinput[indx] = X[j]
                    indx += 1

            self.parzensWindow(xinput, i)
        state = True
        for c in range(epochs):
            self.past_errors = self.errors
            self.errors = 0;
            for i in range(X.shape[0]):  # all samples
                for j in range(self.outClasses.shape[0]):  # for all ten classes # i need to seperate this all for each class
                    x = X[i] * self.weight
                    idx = self.find_closest(self.window, x)
                    self.outClasses[j] = self.pdfs[j, idx]
                mostActivated = np.argmax(self.outClasses)
                actual = y[i, 0]
                if mostActivated != actual:
                    self.errors = self.errors + 1 # (actual-mostActivated)


            percent = self.errors/X.shape[0]
            print("weight "+ str(self.weight) + " contains errors: " + str(self.errors) + " percent: " + str(percent))
            if self.errors > self.past_errors:
                if state == True:
                    state = False
                else:
                    state = False

            if state == True:
                self.weight = self.weight + (nu*(self.errors)) #/X.shape[0]
            else:
                self.weight = self.weight - (nu*(self.errors))

            if self.errors == 0:
                break

        df = pandas.DataFrame(self.pdfs)
        df.to_csv(('parzens/'+str(self.inputNum)+'.csv')) '''






#pnn = ParzensNN(X, labels, 10, 0.00000001)  # 0.000015
#pnn.test(data)

class Node:
    def __init__(self, inputNum_, sig_):
        self.inputNum = inputNum_
        self.mostActivated = 0
        self.errors = 0
        self.past_errors = 0
        self.weight = 1.00
        self.sig = sig_
    def fit(self, nClasses, X):
        self.outClasses = np.zeros(shape=nClasses)  # 10 possible classes not one because your training is based on many classes coming in.
        self.windowRange = [(1.5 * np.amin(X)), (1.5 * np.amax(X))]
        self.stepsize = (self.windowRange[1] - self.windowRange[0]) / (500)
        self.window = np.arange(self.windowRange[0], self.windowRange[1], self.stepsize)
        self.windowSize = self.window.shape[0]
        self.pdfs = np.zeros(shape=(nClasses, self.window.shape[0]))  # creates 10 by windowsize making 10 parzens windowsfor each input

        for i in range(nClasses):
            indx = 0
            xinput = np.zeros(shape=(X.shape[1]))
            for j in range(X.shape[1]):  # shape is enough since each input will be number of samples by num of classes 4*67
                xinput[indx] = X[i, j]
                indx += 1
            self.parzensWindow(xinput, i)
        #state = True

    def parzensWindow(self, X, id):  # X is size 68
        # fill in the values that make up the range covered in window
        for j in range(self.windowSize):  # all of window
            pxw1 = 0.00
            for r in range(X.shape[0]):  # all samples for trainnig data found for that id
                w = self.window[j]
                x = X[r]  # put an if statement
                dif = w - x
                egain = math.exp(float((-(dif * dif)) / (2 * (self.sig * self.sig))))
                u = (1 / (math.sqrt(2 * math.pi) * (self.sig))) * egain
                pxw1 += u
            pxw1 = pxw1 / self.windowSize
            self.pdfs[id, j] = pxw1

        # normalizing parzens
        self.prePdfSum = 0.00
        for j in range(self.windowSize):
            self.prePdfSum += self.pdfs[id, j]
        #print(str(self.prePdfSum))

        for j in range(self.windowSize):
            self.pdfs[id, j] = self.pdfs[id, j] / self.prePdfSum

        df = pandas.DataFrame(self.pdfs)
        df.to_csv(('parzens/' + str(self.inputNum) + '.csv'))

    def test(self, x):
        for i in range(self.outClasses.shape[0]):           # for all classes
            idx = self.find_closest(self.window, x)         # since they are floats need to find closest ones
            self.outClasses[i] = self.pdfs[i, idx]
        self.mostActivated = np.argmax(self.outClasses)     # each node will have one most active class. gets class most activated

    #  since the values are floats find best watch in window
    def find_closest(self, A, target):
        #  A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx


class ParzensNN: # pass in X = 68 * (4* 67), 68 is one for each feature, 67 is distances per feature, 4 is the number class
    def __init__(self, X, num_nodes_, max_sigma_, min_sigma_,  nu_): # X: training data, y: labels, num_nodes: the amount of nodes in network, nu_: learning rate
        self.num_nodes = num_nodes_
        self.nu = nu_
        self.sigStep = (max_sigma_ - min_sigma_) / num_nodes_
        self.sigs = np.arange(min_sigma_, max_sigma_, self.sigStep)
        self.sig = 0.001

        self.nodes = []
        for i in range(self.num_nodes): # 68
            xinput = X[i,:,:]
            self.nodes.append(Node(i, self.sig))
            self.nodes[i].fit(xinput.shape[0],xinput)

    def test(self, x): # x will be 68 by 67 for each frame from cam
        self.classified = np.zeros(shape=(x.shape[0], x.shape[1])).astype(int)
        self.classified2 = np.zeros(shape=(x.shape[0]))
        for i in range(x.shape[0]):  # 68
            for j in range(x.shape[1]):  # 67
                self.nodes[i].test(x[i, j])
                self.classified[i, j] = self.nodes[i].mostActivated

            counts = np.bincount(self.classified[i, :])
            self.classified2[i] = np.argmax(counts)
        #df1 = pandas.DataFrame(self.classified2)
        #df1.to_csv("classificationResults.csv")
        #print(str(self.classified2))
        result = np.bincount(self.classified2.astype(int))
        #result = np.argmax(result)
        print(result)
