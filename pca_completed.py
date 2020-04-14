from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv("Iris.csv")
print (df.head(2))
print (df.info())
labels = df['Species']
print("labels:\n",labels)
X = df.drop(['Id','Species'],axis=1)
print("X:\n",X)

X_std = StandardScaler().fit_transform(X)
#print("X_std:\n",X_std)



pca = PCA(n_components=4)
print("pca=\n",pca)
X_transform = pca.fit_transform(X_std) #voir ça sert à quoi
print ("X_transform\n",X_transform)

explained_var = pca.explained_variance_ratio_
print ("explained_var\n",explained_var)
for var in explained_var:
	print (var)
	plt.bar([1,2,3,4],explained_var,label=var)
	plt.xlabel("Component #")
	plt.ylabel("% Variance Contribution")
	plt.legend()
plt.show()

print ("X_transform[0]\n",X_transform[0])
pca1 = list(zip(*X_transform))
pca1=pca1[0]
print ("pca1\n",pca1)
pca2 = list(zip(*X_transform))
pca2=pca2[1]
print ("pca2\n",pca2)

color_dict = {}

color_dict["Iris-setosa"] = "green"
color_dict["Iris-versicolor"]='red'
color_dict["Iris-virginica"] = 'blue'

i=0
print ("labels.values\n",labels.values)
for label in labels.values:
	plt.scatter(pca1[i],pca2[i],color=color_dict[label])
	i=i+1
print("i_final\n",i)
plt.show()	


