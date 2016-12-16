import numpy as np
import math
def forwardMultiplyGate(x,y):

    return x*y

x = -2 
y = 3

tweak_amount = 1

best_out = -math.inf 

best_x = x
best_y = y

for i in range(0,1000):
    x_try = x + tweak_amount * (2*np.random.random()-1)
    y_try = y + tweak_amount * (2*np.random.random()-1)
    out = forwardMultiplyGate(x_try , y_try )

    if out>best_out:

        best_out = out
        best_x = x_try
        best_y = y_try


print(x_try,y_try,out)

