import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

categories = ['comp.graphics','comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#----------------------------------Generating Histogram of Number of Documents per Class for the 8 Classes--------------------------------------------
index = np.arange(8)
values = []
for i in range(len(twenty_train.target_names)):
    values.append((twenty_train.target == i).sum())


bar_width = 0.75
color = ['b', 'g', 'r', 'c', 'pink',  'm', 'y', 'k']
bars = plt.barh(index, values, bar_width,alpha = 0.8, color = color, align="edge")
plt.xlabel('Number of Documents', fontweight="bold", )
plt.ylabel('Classes', fontweight="bold")
plt.title('Number of Training Documents Per Class', fontweight="bold")
plt.yticks(index + bar_width/2, ('comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'))
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------