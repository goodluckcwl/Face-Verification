import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '/home/chenweiliang/face-verification-python/libsvm-3.21/python')
from svmutil import *
from sklearn.decomposition import PCA

data = sio.loadmat('lfw_feats_sphereface6_iter_50000.mat')
F1 = data['F1'].astype('float64')
F2 = data['F2'].astype('float64')
# 10-folders cross validation
same_label = np.ones(6000)
same_label[3000:6000] = 0

# Normalization
for i in range(len(F1)):
    F1[i, :] = F1[i, :] / np.linalg.norm(F1[i, :])
    F2[i, :] = F2[i, :] / np.linalg.norm(F2[i, :])
thresh = np.zeros(len(F1))
for i in range(len(F1)):
    thresh[i] = np.sqrt( np.sum( ( F1[i, :] - F2[i, :] )*( F1[i, :] - F2[i, :] ) ) )


plt.hist(thresh[0:3000], 100)
plt.hist(thresh[3001:-1], 100)
plt.show()

MAX = np.max(thresh)
MIN = np.min(thresh)
roc_x = []
roc_y = []
for t in np.arange(MIN, MAX, 0.01):
    positive = np.nonzero(thresh<=t)[0]
    negtive = np.nonzero(thresh>t)[0]
    FP = np.nonzero(positive>=3000)[0]
    FPR = len(FP)*1.0/3000
    TP = np.nonzero(positive<3000)[0]
    TPR = len(TP)*1.0/3000
    roc_x.append(FPR)
    roc_y.append(TPR)
plt.plot(roc_x, roc_y)
plt.show()

accuracy = np.zeros(10)
for i in range(10):
    print i
    test_idx = np.hstack( (np.arange(i * 300, (i+1)*300 ,1),
                         np.arange(i * 300 + 3000, (i+1)*300+3000, 1) )
                         )
    train_idx = np.arange(0, 6000, 1).tolist()
    [train_idx.remove(x) for x in test_idx]
    train_idx = np.array(train_idx)

    train = np.vstack((F1[train_idx, :], F2[train_idx, :]))

    # Perform PCA
    # pca = PCA(n_components=512)
    # pca.fit(train)
    # F1_pca = pca.transform(F1)
    # F2_pca = pca.transform(F2)
    # for j in range(len(F1_pca)):
    #     thresh[j] = np.sqrt(np.sum((F1_pca[j, :] - F2_pca[j, :]) * (F1_pca[j, :] - F2_pca[j, :])))

    #
    cmd = ' -t 0 -h 0 -b 1';
    tr_label = same_label[train_idx]
    tr_label = tr_label.tolist()
    tr_thresh = thresh[train_idx]
    tr_thresh = tr_thresh.tolist()
    for j, x in enumerate(tr_thresh):
        tr_thresh[j] = [x]

    model = svm_train(tr_label, tr_thresh, cmd)
    svm_predict(tr_label, tr_thresh, model)
    test_label = same_label[test_idx]
    test_label = test_label.tolist()
    test_thresh = thresh[test_idx]
    test_thresh = test_thresh.tolist()
    for j, x in enumerate(test_thresh):
        test_thresh[j] = [x]
    cla, acc, deci = svm_predict(test_label, test_thresh, model, '-b 1')
    accuracy[i] = acc[0]

print 'Mean accuracy is %f' % (np.mean(accuracy))
