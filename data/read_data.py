import pickle
df=open('/Users/chenqiaoling/Desktop/blog/codes/LinS/data/dump_data_all_gahter.pickle','rb')

data=pickle.load(df)
print(data)