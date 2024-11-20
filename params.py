class params:
    def __init__(self,mu,a1,a2,ep,lmbda,gamma,v0,u0,a0ic,a0cur):
        self.mu = mu
        self.a1 = a1
        self.a2 = a2
        self.ep = ep
        self.lmbda = lmbda
        self.gamma = gamma
        self.v0 = v0
        self.u0 = u0
        self.a0ic = a0ic
        self.a0cur = a0cur
    
    def setA0cur(self, a0):
        self.a0cur = a0
    