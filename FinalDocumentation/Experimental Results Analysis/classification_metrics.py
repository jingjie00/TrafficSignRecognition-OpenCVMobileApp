#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np


# In[8]:


svm_results = np.genfromtxt('SVM_res.csv')
rf_results = np.genfromtxt('Random_Forest_res.csv')
truth = np.genfromtxt('testLabels.csv')


# In[14]:


from sklearn.metrics import accuracy_score

test_acc = accuracy_score(truth, rf_results)
test_acc_svc = accuracy_score(truth, svm_results)

print("RF Testing Accuracy: {:.4f}".format(test_acc))
print("SVM Testing Accuracy: {:.4f}".format(test_acc_svc))


# In[15]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cmocean

labels = [1,2,3,4,5,6]

cm = confusion_matrix(truth, rf_results)
print(cm)

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap = cmocean.cm.dense)

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{}'.format(z), ha='center', va='center', color='red')
    
plt.title('Confusion matrix of Random Forest')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[11]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cmocean

labels = [1,2,3,4,5,6]

cm = confusion_matrix(truth, svm_results)
print(cm)

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap = cmocean.cm.dense)

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{}'.format(z), ha='center', va='center', color='red')
    
plt.title('Confusion matrix of Linear SVM')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[12]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('precision = ', precision_score(truth, rf_results, average='micro'))
print('recall = ', recall_score(truth, rf_results, average='micro'))
print('f1 score = ', f1_score(truth, rf_results, average='micro'))


# In[13]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('precision = ', precision_score(truth, svm_results, average='micro'))
print('recall = ', recall_score(truth, svm_results, average='micro'))
print('f1 score = ', f1_score(truth, svm_results, average='micro'))

