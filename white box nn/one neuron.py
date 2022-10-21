import numpy as np

def accuracy(ytst,ypred):
    print("mean error:",np.mean(np.abs(ypred-ytst)))
    print("rmse:",np.sqrt(((ypred-ytst)**2).mean()))
    
#importing data
a=np.genfromtxt(r'C:\Users\karu0\OneDrive\Documents\STUDY_MATERIALS,SUBMISSIONS(2022)\AI\WineQT.csv',delimiter = ',')
a=np.delete(a,0,axis=0)


# train test split (adding ones column for numpy method)
a=a[~np.isnan(a).any(axis=1),:]
x=a[:,:-2]
x=np.append(np.ones((len(a),1)),x,axis=1) #adding ones column in input for bias
y=a[:,-2:-1]
indices = np.random.permutation(len(a))
size = round(len(a)*0.7)
np.random.shuffle(indices)
training_idx, test_idx = indices[:size], indices[size:]
xtr, xtst = x[training_idx,:], x[test_idx,:]
ytr, ytst = y[training_idx,:], y[test_idx,:]


#======using numpy alone======
#normal method
normal = lambda x,y: (np.linalg.inv(x.T @ x)) @ (x.T @ y)
w=normal(x,y)
predict = lambda x,w: x @ w
ypred=predict(xtst,w)
print("\nnormal accuracy")
accuracy(ytst,ypred)

#gradient descent method
w1=np.zeros((len(x[0]),1))
step = lambda x,y,w: x.T @ (predict(xtr,w) - y) *(0.00001/len(x))

'''print(step)

for i in range(1000):
    w1-=step(xtr,ytr,w1)
    print(w1)'''
    

ypred1=predict(xtst,w1)
print("\ngradient accuracy")
accuracy(ytst,ypred1)

#======using sklearn======
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
x=a[:,:-2]
x=np.append(np.ones((len(a),1)),x,axis=1) #adding ones column in input for bias
y=a[:,-2:-1]
indices = np.random.permutation(len(a))
size = round(len(a)*0.7)
np.random.shuffle(indices)
training_idx, test_idx = indices[:size], indices[size:]
xtr, xtst = x[training_idx,:], x[test_idx,:]
ytr, ytst = y[training_idx,:], y[test_idx,:]
reg.fit(xtr, ytr)
ypred2 = reg.predict(xtst)
print("\nscikit learn accuracy")
accuracy(ytst,ypred2)

print(np.append(w,w1,axis=1))
