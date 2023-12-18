import pandas as pd
import matplotlib.pyplot as plt		#用于制图

dataFram1_train = pd.read_csv('data\\UVAdataset_csv\\dataset_mult_tr.CSV')
data1_value_train = dataFram1_train.values

X1_value_train = data1_value_train[:,:-1].astype(float)
Y1_value_train = data1_value_train[:,-1:].astype(int)

for i in range(54):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.scatter(X1_value_train[:, i], Y1_value_train, s=5)  # 横纵坐标和点的大小
plt.show() 
