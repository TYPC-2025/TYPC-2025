import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

"""数据集读取"""
from sklearn.datasets import  fetch_openml
mnist = fetch_openml('mnist_784',version=1)
X, y = mnist["data"].values, mnist["target"].values.astype(np.int8)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""洗牌操作"""
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
print(shuffle_index)

"""交叉验证"""
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(y_train_5[:10])

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter = 5,random_state=42) #实例化
sgd_clf = sgd_clf.fit(X_train,y_train_5)
print(sgd_clf.predict([X[35000]])) # 用分类器来预测一下它的结果
print(y[35000])
# 最简便方法(应该掌握这种写法)
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')
print(cross_val) # 后面步骤可以是求平均值
# 使用函数进行切分
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))

"""混淆矩阵[[TN,FP]
           [FN,TP]]"""
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)
print(y_train_pred.shape)
print(X_train.shape)

from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_train_5, y_train_pred)
print(conf_mx)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
# F1-score = 2 / (1/precision + 1/recall)
from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))

"""阈值（阈值较低，得到精度较低，召回率较高）"""
y_scores = sgd_clf.decision_function(X_train) # 修改为对整个训练集计算决策分数
print(y_scores.shape) # 添加打印以确认形状
print(y_scores) # 负值：表示该样本被预测为 负类
# 设置阈值
t = 250000
y_pred = (y_scores > t)
print(y_pred)
"""ROC曲线直接生成阈值"""
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

# 通过AUC的值进行模型评估（最好情况是AUC = 1）
# 纯随机分类器的AUC = 0.5
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5, y_scores))