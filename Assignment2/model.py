import torch.nn as nn

class Neural_Net(nn.Module):

    # input layer, 3 hidden layers and a output layer

    def __init__(self,n_x,n_h1,n_h2,n_h3,n_y):

        super(Neural_Net,self).__init__()
        self.l1 = nn.Linear(n_x,n_h1)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(n_h1,n_h2)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(n_h2,n_h3)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(n_h3,n_y)
        self.network = nn.Sequential(
            
                        # first hidden layer
                        self.l1,self.a1,

                        nn.BatchNorm1d(n_h1),

                        # second hidden layer
                        self.l2,self.a2,

                        nn.BatchNorm1d(n_h2),
                        
                        # third hidden layer
                        self.l3,self.a3,

                        nn.BatchNorm1d(n_h3),

                        # output layer
                        self.l4
                        
                      )
        
    def forward(self,x):
        output = self.network(x)
        return output