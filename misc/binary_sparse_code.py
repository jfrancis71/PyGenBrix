import numpy as np


class BinarySparseCode():
    def __init__(self, k=100, n=1000, p_x=.002, f=.3):
        self.k = k
        self.n = n
        self.p_x = p_x
        self.codebook = np.random.binomial(n=1, p=.1, size=[self.k, self.n])
        arr = np.zeros([self.k])
        arr[:int(f*self.k)] = 1
        self.codebook = np.transpose(np.array([np.random.permutation(arr) for _ in range(self.n)]))

    def p_c_given_x(self, codeword, pos, value):
        possible_x = np.ones([self.n])
        if value == 0:
            possible_x[pos] = 0
        p = 1.0
        for c in range(self.k):
            if codeword[c] == 0:
                p = p*(1-self.p_x)**(self.codebook[c]*possible_x).sum()
                if value == 1 and self.codebook[c,pos] == 1:
                    p = 0.0
            else:
                p = p*(1-(1-self.p_x)**(self.codebook[c]*possible_x).sum())
            if codeword[c] == 0:
                possible_x = possible_x*(1-self.codebook[c])
        return p

    def marginal_x(self, codeword, pos):
        return (self.p_x * self.p_c_given_x(codeword, pos, 1) /
            ((1-self.p_x) * self.p_c_given_x(codeword, pos, 0) + self.p_x * self.p_c_given_x(codeword, pos, 1)))
