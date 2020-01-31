import torch.nn as nn

__all__ = ['LinearRegression_t', 'LogisticRegression_t', 'SoftmaxRegression_t']

class LinearRegression_t(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression_t, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class LogisticRegression_t(LinearRegression_t):
    def __init__(self, in_features, out_features):
        super(LogisticRegression_t, self).__init__(in_features, out_features)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, x):
        out = super().forward(x)
        return self.sigmoid(out)

class SoftmaxRegression_t(LogisticRegression_t):
    def __init__(self, in_features, out_features):
        super(SoftmaxRegression_t, self).__init__(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = super().forward(x)
        return self.softmax(out)

        
    

