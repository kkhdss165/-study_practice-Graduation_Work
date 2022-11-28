import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

xy = [[0,1], [1,2], [6,37]]

a = np.array(xy)
x = np.arange(0,6,1)


print(xy,x)

intrp = interp1d(a[:,0], a[:,1],  kind='quadratic')
y = intrp(x)
print(y)

plt.plot(x,y, marker="X")
plt.show()

