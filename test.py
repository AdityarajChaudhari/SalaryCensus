import pickle
import numpy as np

model = pickle.load(open('./ModelSaving/dtc.pkl', 'rb'))
scalar = pickle.load(open('./Scaler/Scalar.pkl', 'rb'))

s = scalar.transform([[39, 77516, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]])
print(s)
a = (model.predict(s))
print(a)
print(type(a))
