# coding:utf-8

from pylab import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

# 加载人脸数据集
data = fetch_olivetti_faces()
targets = data.target

data = data.images.reshape((len(data.images), -1))
train = data[targets < 30]
test = data[targets >= 30]  # 在独立的人上进行测试

# 在人群子集上进行测试
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
X_train = train[:, :np.ceil(0.5 * n_pixels)]  # 上半部分人脸
y_train = train[:, np.floor(0.5 * n_pixels):]  # 下半部分人脸
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]

# 拟合估测器
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}
ESTIMATORS_zh = {
    "Extra trees":u"随机树",
    "K-nn": u"K近邻",
    "Linear regression":u"线性回归" ,
    "Ridge": u"岭回归",

}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# 绘制完成的人脸

myfont = matplotlib.font_manager.FontProperties(fname="Microsoft-Yahei-UI-Light.ttc")
print (myfont)
mpl.rcParams['axes.unicode_minus'] = False

image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle(u"采用多输出估测器进行人脸完成", size=16,fontproperties=myfont)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    if not i:
       sub.set_title(u"真实人脸",fontproperties=myfont)
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        if not i:
            sub.set_title(ESTIMATORS_zh[est],fontproperties=myfont)
        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")
plt.show()