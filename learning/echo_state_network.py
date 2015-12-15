from numpy import *
from matplotlib.pyplot import *
import scipy.linalg

class EchoStateNetwork():

    inSize = 1         # Number of inputs
    outSize = 1        # Number of outputs
    resSize = 1000     # Size of reservoir.
    a = 0.8              # Leaking rate
    w_back = False

    # load the data
    trainLen = 2000
    testLen = 2000
    initLen = 0

    def initializeNetwork(self, inputs, outputs, reservoir, back=False):
        self.inSize = inputs
        self.outSize = outputs
        self.resSize = reservoir
        self.w_back = back

        random.seed(42)
        self.Win = (random.rand(self.resSize,1+self.inSize)-0.5) * 1
        self.W = random.rand(self.resSize,self.resSize)-0.5
        if self.w_back:
            self.Wback = (random.rand(self.resSize,self.outSize)-0.5) * 1

        print 'Computing spectral radius...',
        self.rhoW = max(abs(linalg.eig(self.W)[0]))
        print 'done.'
        self.W *= 1.25 / self.rhoW

    def trainNetwork(self, training_input, training_output):
        self.trainLen = len(training_input)

        # allocated memory for the design (collected states) matrix
        X = zeros((self.trainLen-self.initLen, 1+self.inSize+self.resSize))
        # set the corresponding target matrix directly
        Yt = array(training_output[self.initLen:self.trainLen])

        # run the reservoir with the data and collect X
        self.x = zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = array(training_input[t])
            if self.w_back:
                if t > 0:
                    y_prev = array(training_output[t-1])
                else:
                    y_prev = array([0]*self.outSize)
                self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack(insert(u,0,1)) ) + dot( self.W, self.x ) + dot( self.Wback, vstack(y_prev) ))
            else:
                self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack(insert(u,0,1)) ) + dot( self.W, self.x ) )
            if t >= self.initLen:
                X[t-self.initLen,:] = hstack(insert(insert(self.x, 0, u), 0, 1))


        # train the output
        # reg = 1e-8  # regularization coefficient
        X_p = linalg.pinv(X)
        self.Wout = np.transpose(dot( X_p, Yt ))

    def testNetwork(self, test_input, test_output):
        # Note: assumption that x is
        self.testLen = len(test_input)
        Y = []

        for t in range(self.testLen):
            u = array(test_input[t])
            if self.w_back:
                if t > 0:
                    y_prev = array(test_output[t-1])
                else:
                    y_prev = array([0]*self.outSize)
                self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack(insert(u,0,1)) ) + dot( self.W, self.x ) + dot( self.Wback, vstack(y_prev) ))
            else:
                self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack(insert(u,0,1)) ) + dot( self.W, self.x ) )
            y = dot( self.Wout, hstack(insert(insert(self.x, 0, u), 0, 1)) )
            Y.append(y.tolist())

        return Y