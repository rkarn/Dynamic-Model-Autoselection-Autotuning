from matplotlib.pyplot import * 
import matplotlib.pyplot as plt
import pylab
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.cluster import KMeans
pylab.rcParams['figure.figsize'] = (16.0, 5.0)
df = pd.read_csv("UNSW_NB15_training-set-formated.csv")
import sklearn
 
X=df.values[:,:-1]
y=df.values[:,-1]

print "Lables are of type",set(y.tolist())
from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(X)
pca_3d = pca.transform(X)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')

fig6 = plt.figure()
ax6 = fig5.add_subplot(111, projection='3d')

fig7 = plt.figure()
ax7 = fig7.add_subplot(111, projection='3d')

fig8 = plt.figure()
ax8 = fig8.add_subplot(111, projection='3d')

fig9 = plt.figure()
ax9 = fig9.add_subplot(111, projection='3d')

fig10 = plt.figure()
ax10 = fig10.add_subplot(111, projection='3d')

for i in range(0, pca_3d.shape[0]):
    if y[i] == 'Fuzzers':
        ax1.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='r', marker='^')
        ax1.set_xlabel('PCA1')
        ax1.set_ylabel('PCA2')
        ax1.set_zlabel('PCA3 Fuzzers')
        
    elif y[i] == 'Exploits':
        ax2.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='g', marker='^')
        ax2.set_xlabel('PCA1')
        ax2.set_ylabel('PCA2')
        ax2.set_zlabel('PCA3 Exploits')
        
    elif y[i] == 'Normal':
        ax3.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='b', marker='^')
        ax3.set_xlabel('PCA1')
        ax3.set_ylabel('PCA2')
        ax3.set_zlabel('PCA3 Normal')
        
    elif y[i] == 'Generic':
        ax4.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='c', marker='^')
        ax4.set_xlabel('PCA1')
        ax4.set_ylabel('PCA2')
        ax4.set_zlabel('PCA3 Generic')


    elif y[i] == 'Worms':
        ax5.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='y', marker='^')
        ax5.set_xlabel('PCA1')
        ax5.set_ylabel('PCA2')
        ax5.set_zlabel('PCA3 Worms')
    
    elif y[i] == 'Analysis':
        ax6.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='m', marker='^')
        ax6.set_xlabel('PCA1')
        ax6.set_ylabel('PCA2')
        ax6.set_zlabel('PCA3 Analysis')
        
    elif y[i] == 'Backdoor':
        ax7.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='k', marker='^')
        ax7.set_xlabel('PCA1')
        ax7.set_ylabel('PCA2')
        ax7.set_zlabel('PCA3 Backdoor')
    
    elif y[i] == 'DoS':
        ax8.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='r', marker='.')
        ax8.set_xlabel('PCA1')
        ax8.set_ylabel('PCA2')
        ax8.set_zlabel('PCA3 DoS')
    
    elif y[i] == 'Reconnaissance':
        ax9.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='g', marker='.')
        ax9.set_xlabel('PCA1')
        ax9.set_ylabel('PCA2')
        ax9.set_zlabel('PCA3 Reconnaissance')
        
    elif y[i] == 'Shellcode':
        ax10.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c='b', marker='.')
        ax10.set_xlabel('PCA1')
        ax10.set_ylabel('PCA2')
        ax10.set_zlabel('PCA3 Shellcode')
        
plt.show()
