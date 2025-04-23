import os
import numpy as np

np.random.seed(0)

with open("data.txt", "w") as f:
    for i in range(500):
        for j in range(50):
            integer = np.random.randint(1,500)
            f.write(f"{np.base_repr(integer,2)}->{integer}")
            if j < 49:
                f.write(",")
        f.write("\n")
    