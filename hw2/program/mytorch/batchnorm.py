# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):
        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        self.x = x

        # Inference mode
        if eval:
            self.mean = None
            self.var = None
            self.norm = None
            self.out = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps) * self.gamma + self.beta
            return self.out

        # Training mode

        self.mean = np.mean(self.x, axis=0)
        self.var = np.var(self.x, axis=0)

        self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.norm * self.gamma + self.beta

        # update running batch statistics
        self.running_mean = self.running_mean * self.alpha + self.mean * (1 - self.alpha)
        self.running_var = self.running_var * self.alpha + self.var * (1 - self.alpha)

        return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        m = self.x.shape[0]  # batch size
        dgamma = np.sum(delta * self.norm, axis=0)
        dbeta = np.sum(delta, axis=0)
        self.dgamma[0] = dgamma
        self.dbeta[0] = dbeta

        dx_hat = self.gamma * delta
        dvar = np.sum(dx_hat * (self.x - self.mean), axis=0) * (-1 / 2) * (self.var + self.eps) ** (-3 / 2)
        dmean = -np.sum(dx_hat, axis=0) / np.sqrt(self.var + self.eps) - 2 / m * dvar * np.sum(self.x - self.mean,
                                                                                               axis=0)

        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0 * (self.x - self.mean) / m + dmean / m
        return dx
