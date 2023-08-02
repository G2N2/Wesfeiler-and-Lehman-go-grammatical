import pandas as pd
import numpy as np

name = "MUTAG"

l = []

for i in range(10):
    dt = pd.read_csv("data/"+name+str(i)+"_predonly_GMN.dat")
    l.append(dt['testacc'].max())
l = np.array(l)
print(l.mean(),l.std())

