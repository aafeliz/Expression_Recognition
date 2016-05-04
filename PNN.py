import numpy as np
import math
import pandas

#pnn = ParzensNN(X, labels, 10, 0.00000001)  # 0.000015
#pnn.test(data)

# TODO: implement the Delta errors into the correction step in order to temperaraly accelerate the learning rate
def insert_line(fileName, lineNum, text):
    with open(fileName, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    tempData = data[lineNum] +'\n'
    # now change the 2nd line, note that you have to add a newline
    #data[lineNum] = text
    data[lineNum] = data[lineNum].replace(tempData, text)
    # and write everything back
    with open(fileName, 'w') as file:
        file.writelines(data)

def get_sigma(fileName, sigma):
    with open(fileName, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    sigma = data[sigma]
    return sigma

class Node:
    def __init__(self, inputNum_, sig_, epoch_, nu_):
        self.inputNum = inputNum_
        self.mostActivated = 0
        self.errors = 0
        self.past_errors = 0
        self.weight = 1.00
        self.sig = sig_
        self.epoch = epoch_
        self.nu = nu_
        self.gradient = "decending"
        self.preDeltaErrors = 0
        self.deltaErrors = 0

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
        # **************************************** CORRECTION STEP *****************************************
        #'''
        state = True
        self.errors = 0
        epochNum =0
        for c in range(self.epoch):
            self.preDeltaErrors = self.deltaErrors
            self.past_errors = self.errors
            self.errors = 0
            for q in range(X.shape[0]):
                for i in range(X.shape[1]):  # all samples-- index 1 since shape is 4* 67
                    self.test(X[q, i])
                    actual = q # y[i, 0]
                    if self.mostActivated != actual:
                        self.errors = self.errors + 1  # (actual-mostActivated)
            self.percent = self.errors / X.shape[0]

            print("sigma " + str(self.sig) + " contains errors: " + str(self.errors) + " percent: " + str(self.percent) + "on epoch: " + str(epochNum))
            self.deltaErrors = abs(self.past_errors - self.errors)
            if self.errors > self.past_errors:
                if state is True:
                    state = False
                else:
                    state = True

            if state is True:
                self.sig = self.sig + (self.nu * (self.percent))  # self.weight instead of sig and self.errors instaead of self.percent
                self.gradient = "accending"
            else:
                self.sig = self.sig - (self.nu * (self.percent))
                self.gradient = "decending"

            if self.errors == 0:
                break
            # rebuild pdfs with the new sigma
            for i in range(nClasses):
                indx = 0
                xinput = np.zeros(shape=(X.shape[1]))
                for j in range(
                        X.shape[1]):  # shape is enough since each input will be number of samples by num of classes 4*67
                    xinput[indx] = X[i, j]
                    indx += 1
                self.parzensWindow(xinput, i)
            epochNum = epochNum + 1
        # the weights of the output will depend on the percent error each node generates
        self.weight = self.percent
        #'''
        # **************************************** CORRECTION STEP END *****************************************
        if self.inputNum < 49:
            insert_line('sigmas.txt', self.inputNum, str(self.sig))
        else:
            insert_line('sigmas2.txt', self.inputNum-49, str(self.sig))
        df = pandas.DataFrame(self.pdfs)
        df.to_csv(('parzens/' + str(self.inputNum) + '.csv'))

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

        #df = pandas.DataFrame(self.pdfs)
        #df.to_csv(('parzens/' + str(self.inputNum) + '.csv'))

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
        self.sig = 0.0001
        self.result = [0, 0, 0, 0]
        '''for i in range(99):
            insert_line('sigmas.txt', i, str(self.sig))'''

        self.nodes = []
        for i in range(self.num_nodes):  # 68
            xinput = X[i, :, :]
            tempSig = 0.00
            if i < 49:
                tempSig = float(get_sigma('sigmas.txt', int(i)))
            else:
                tempSig = float(get_sigma('sigmas2.txt', int(i-49)))
            self.nodes.append(Node(i, tempSig, 6, self.nu))
            self.nodes[i].fit(xinput.shape[0], xinput)

    def test(self, x): # x will be 68 by 67 for each frame from cam
        self.classified = np.zeros(shape=(x.shape[0], x.shape[1])).astype(int)
        self.classified2 = np.zeros(shape=(x.shape[0]))
        weights = []
        for i in range(len(self.nodes)):
            weights.append(self.nodes[i].weight)

        for i in range(x.shape[0]):  # 68
            for j in range(x.shape[1]):  # 67
                self.nodes[i].test(x[i, j])
                self.classified[i, j] = self.nodes[i].mostActivated
            counts = np.bincount(self.classified[i, :])
            self.classified2[i] = np.argmax(counts)
        #df1 = pandas.DataFrame(self.classified2)
        #df1.to_csv("classificationResults.csv")
        print(str(self.classified2))
        self.result = np.bincount(self.classified2.astype(int), weights=weights)
        allSum = sum(self.result[:])
        self.result = (self.result/allSum)
        self.result = self.result*100
        self.result = self.result.astype(int)
        #result = np.argmax(result)
        print(self.result)
