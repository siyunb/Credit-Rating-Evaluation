########读取台湾信贷数据，数据预处理
library(MASS)
taiwan<- read.csv("D:/bigdatahw/算法/作业/card1.csv",head=TRUE,sep=',') #读取台湾信用数据
taiwan=taiwan[1:4000,-1]      #删除第一个变量ID变量,样本量太大，在本地无法运行，所以抽取前4000个变量
table(taiwan$default.payment.next.month)    #查看下月是否违约的报告，违约比率与截取数据前相差不大
library(mice)
md.pattern(taiwan)  #生成缺失报告,数据完整没有缺失值
set.seed(1234)      #设置随机种子，可以重复实验结果
taiwan$default.payment.next.month=as.factor(taiwan$default.payment.next.month)  #将是否违约转化为因子型变量
train=sample(1:nrow(taiwan), round(0.8* nrow(taiwan)))  #抽取80%,24000个数据作为训练集

######Bagging集成方法，利用决策树作为基分类器
library(randomForest)
taiwan_bag=randomForest(default.payment.next.month~., taiwan[train, ], mtry=round(sqrt(ncol(taiwan)),0)) 
#所构造的基分类器的决策树采用变量为所有变量的p^1/2个。  

###将Bagging模型用于测试集
taiwan_bag.pred = predict(taiwan_bag, taiwan[-train,-24])  #去掉训练集数据，并同时去掉预测变量
accuracy=mean(taiwan_bag.pred==taiwan[-train,24]) * 100    #计算准确率,正确率达到79.25%
print(sprintf('The bagging accuracy is %f', accuracy))

######随机森林
###构造随机森林模型。随机树也就是bagging+决策树
taiwan_rf=randomForest(default.payment.next.month~., taiwan[train,],importance=T) #显示变量重要性
###测试
taiwan_rf.pred = predict(taiwan_rf, taiwan[-train,])        #选取树的特征变量和树的棵数都是默认的
accuracy=mean(taiwan_rf.pred==taiwan[-train,24]) * 100      #计算准确率,正确率达到80.25%
print(sprintf('The randomforest accuracy is %f', accuracy)) #与把bagging类似，只是提高了树的棵数，使得准确率有所提高


######Boosting,用gbm包通过集成来提升决策树能力
###建立Boosting模型
library(gbm)            #载入gbm包
#采用梯度提升方法Gradient Boosting
taiwan_boost=gbm(as.numeric(as.character(default.payment.next.month))~., taiwan[train,], distribution='bernoulli', 
               n.trees=5000, interaction.depth=4)         
#基分类器仍为决策树，形成的基分类器为5000个，也就是迭代次数为5000，
#使用修正测试点的分布，也就是损失函数为伯努利分布，分类问题一般选择bernoulli分布
#每棵树的做大深度为4减轻过拟合
#注意二分变量必须为0,1,且为数值型
taiwan_boost.pred = predict(taiwan_boost, taiwan[-train,-24], n.trees=5000)
#建立混淆矩阵
confusion(as.numeric(as.character(taiwan[-train,]$default.payment.next.month)) > 0, taiwan_boost.pred > 0)
#如混淆矩阵所示，测试错误率为23.84%，测试集的正确率为76.16%


#还可以尝试下adaboost，同样是利用gbm包，采用自适应提升方法AdaBoost

#首先需要进行xgboost包的安装，只需在R中运行install.packages(‘xgboost’)即可。安装好包后，我们对乳腺癌数据进行建模预测。
library('xgboost')
x<-taiwan[train,1:23]
y<-taiwan[train,24]
x<-apply(x,2,as.numeric)               #将x的列变量转化为数字型
y<-as.numeric(y)-1              #将y也转化为数字型变量
bst<-xgboost(data=x,label=y,max.depth=2,eta=1,nround=10,objective='binary:logistic')
#在此处，我们选取最大深度为2，学习速率为1，决策树棵树为10，目标函数是基于二分类问题的logistic损失进行模型建立。
# xgboost训练过程
#也可以利用函数xgb.cv对数据进行交叉验证分析，利用交叉验证确定均方误差,利用交叉检验确定最佳迭代次数
cv.res<-xgb.cv(data=x,label=y,max.depth=2,eta=1,nround=10,objective='binary:logistic',nfold=5)

#利用predict函数进行预测
test=taiwan[-train,1:23]
test<-apply(test,2,as.numeric) 
pred<-predict(bst,test)
pred<-round(pred, 0)  #转化为分类变量
table(as.factor(pred),as.factor(taiwan[-train,24]))	 #混淆数量矩阵
accuracy=mean(as.factor(pred)==taiwan[-train,24]) * 100  #预测准确率为81.75%,速度很快
accuracy


# 加载包和数据
library(gbm)
data(PimaIndiansDiabetes2,package='mlbench')
# 将响应变量转为0-1格式
data <- PimaIndiansDiabetes2
data$diabetes <- as.numeric(data$diabetes)
data <- transform(data,diabetes=diabetes-1)
# 使用gbm函数建模
model <- gbm(diabetes~.,data=data,shrinkage=0.01,
             distribution='bernoulli',cv.folds=5,
             n.trees=3000,verbose=F)
# 用交叉检验确定最佳迭代次数
best.iter <- gbm.perf(model,method='cv')