W = 51
F = 3
S = 2
P = 1
import numpy as np

def conv_size(W, F, S, P):
    return (W - F + 2 * P) / S + 1


print(conv_size(W, F, S, P))


W= 28

F=3
S=2
P=1
OP = 1
def transpose_conv_size(W, F, S, P,OP):
    return (W - 1) * S - 2 * P + F +OP

print(transpose_conv_size(W, F, S, P,OP))