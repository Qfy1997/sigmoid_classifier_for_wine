import numpy as np,matplotlib.pyplot as plt,copy
import sklearn.datasets as dts
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import csv

#显示中文和负号
plt.rcParams['axes.unicode_minus']=False
def data_process():
    # 9、加载wine数据集load_wine()
    data=dts.load_wine()
    # 10、获取特征x
    x=data.data
    # 11、获取标签y
    y=data.target
    # 12、标准化特征缩放
    x_mean=np.mean(x,axis=0)
    x_std=np.std(x,axis=0,ddof=1)
    x=(x-x_mean)/x_std
    # 13、初始化，特征添加全为1的第一列
    m,n=x.shape
    x=np.c_[np.ones((m,1)),x]
    y=np.c_[y]
    # 14、标签独热处理(新增)
    one_hot=OneHotEncoder(categories='auto')#自动处理,不会从0-9编码
    y=one_hot.fit_transform(y).toarray()#先训练再应用,toarray转化成矩阵
    # 15、创建随机排列
    np.random.seed(5)
    order=np.random.permutation(m)
    x=x[order]
    y=y[order]
    # 16、划分训练集：测试集=0.6：0.4
    num=int(m*0.6)
    train_x,test_x=np.split(x,[num])
    train_y,test_y=np.split(y,[num])
    return train_x,train_y,test_x,test_y
# 1、要求神经网络模型结构有四层（包括输入层，中间层2层，输出层）
# 2、要求中间层神经元个数满足9,8的结构，完成wine三分类神经网络底层实现
# 3、定义sigmoid函数及其导数
def sigmoid(g,deriy=False):
    if deriy:
        return g*(1-g)
    else:
        return 1.0/(1+np.exp(-g))
# 4、正向传播
def fp(x,theta1,theta2,theta3):
    a1=x
    z2=a1.dot(theta1)
    a2=sigmoid(z2)

    z3=a2.dot(theta2)
    a3=sigmoid(z3)

    z4=a3.dot(theta3)
    a4=sigmoid(z4)
    h=a4
    return a2,a3,h

# 5、根据交叉熵，定义代价函数
def loss_func(h,y,lamda,theta1R,theta2,theta3):
    m=len(h)
    R=lamda/(2*m)*(np.sum(theta1R**2)+np.sum(theta2**2)+np.sum(theta3**2))
    J=-np.mean(y*np.log(h)+(1-y)*np.log(1-h))+R
    return J

# 6、进行反向传播return dt,s,
def bp(x,h,y,a2,a3,theta2,theta3,lamda,theta1R):
    s4=h-y
    s3=s4.dot(theta3.T)*sigmoid(a3,deriy=True)
    s2=s3.dot(theta2.T)*sigmoid(a2,deriy=True)

    dt3=a3.T.dot(s4)+lamda*theta3
    dt2=a2.T.dot(s3)+lamda*theta2
    dt1=x.T.dot(s2)+lamda*theta1R
    return dt3,dt2,dt1

# 7、进行梯度下降
def train_model(x,y,alpha=0.7,iters=100,lamda=0.1):
    xm,xn=x.shape
    ym,yn=y.shape
    loss_list=[]
    theta1=np.random.randn(xn,9)
    theta2=np.random.randn(9,8)
    theta3=np.random.randn(8,yn)
    for i in range(iters):
        a2, a3, h=fp(x,theta1,theta2,theta3)
        theta1R = copy.copy(theta1)
        theta1R[0]=0
        loss=loss_func(h,y,lamda,theta1R,theta2,theta3)
        loss_list.append(loss)
        dt3, dt2, dt1=bp(x,h,y,a2,a3,theta2,theta3,lamda,theta1R)
        theta1=theta1-alpha*(1.0/xm)*dt1
        theta2 = theta2 - alpha * (1.0 / xm) * dt2
        theta3 = theta3 - alpha * (1.0 / xm) * dt3
    return loss_list,theta1,theta2,theta3

# 8、定义精度函数，返回分类准确率
def acc_func(h,y):
    y_label=np.argmax(y,axis=1)
    h_label=np.argmax(h,axis=1)
    acc=np.mean(y_label==h_label)
    return acc

if __name__ == '__main__':
    # 17、训练数据，调整超参数
    train_x, train_y, test_x, test_y=data_process()
    print(train_x[0])
    print(train_y[0])
    loss_list, theta1, theta2, theta3=train_model(train_x,train_y)
    # 18、绘制代价函数图
    # plt.plot(loss_list,c='m')
    # plt.show()
    # 19、输出训练集精度
    a2, a3, train_h=fp(train_x,theta1,theta2,theta3)
    print(train_h)
    print('训练集精度:',acc_func(train_h,train_y))
    # 20、输出测试集精度
    a2, a3, test_h = fp(test_x, theta1, theta2, theta3)
    print(test_h)
    print(test_y)
    print(len(test_h))
    print('测试集精度:', acc_func(test_h, test_y))

    y_label=np.argmax(test_y,axis=1)
    h_label=np.argmax(test_h,axis=1)
    print(y_label)
    print(h_label)
    ptof=[]
    for i in range(len(y_label)):
        if y_label[i]==h_label[i]:
            ptof.append(1)
        else:
            ptof.append(0)
    print(ptof)
    print(len(ptof))
    print(len(y_label))
    # print(len(y_label))
    # print(len(h_label))
    # confusion_matrix=np.zeros((3,3))
    # print(confusion_matrix)
    # for i in range(len(y_label)):
    #             confusion_matrix[y_label[i]][h_label[i]]+=1
    # print(confusion_matrix)
    # # 混淆矩阵(热力图)
    # conf_matrix = pd.DataFrame(confusion_matrix, index=['0', '1', '2'], columns=['0', '1', '2'])  # 数据有3个类别
    # plt.figure()
    # sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues")
    # plt.ylabel('True label', fontsize=14)
    # plt.xlabel('Predict label', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # # plt.savefig('confusion.pdf', bbox_inches='tight')
    # plt.show()

    data=test_h
    pca = PCA(n_components=2)
    pca.fit(data)
    pca_data = pca.transform(data)
    print(data.shape)
    print(pca_data.shape)
    print(pca_data)
    data_final=[]
    for i in range(72):
        data=[]
        data.append(float(pca_data[i][0]))
        data.append(float(pca_data[i][1]))
        if ptof[i]==1:
            data.append("True")
        else:
            data.append("False")
        if y_label[i]==0:
            data.append("category0")
        elif y_label[i]==1:
            data.append("category1")
        else:
            data.append("category2")
        data.append(int(y_label[i]))
        data_final.append(data)
    print(data_final)
    var_ratio = pca.explained_variance_ratio_

    for idx, val in enumerate(var_ratio, 1):
        print("Principle component %d: %.2f%%" % (idx, val * 100))

    with open('pca.csv', 'w', newline='') as file:
        fields = ['pc1', 'pc2','ptof','realabel']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in data_final:
            writer.writerow({'pc1': item[0], 'pc2': item[1],'ptof':item[2],'realabel':item[3]})
        
    

    
