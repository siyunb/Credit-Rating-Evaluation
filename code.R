########��ȡ̨���Ŵ����ݣ�����Ԥ����
library(MASS)
taiwan<- read.csv("D:/bigdatahw/�㷨/��ҵ/card1.csv",head=TRUE,sep=',') #��ȡ̨����������
taiwan=taiwan[1:4000,-1]      #ɾ����һ������ID����,������̫���ڱ����޷����У����Գ�ȡǰ4000������
table(taiwan$default.payment.next.month)    #�鿴�����Ƿ�ΥԼ�ı��棬ΥԼ�������ȡ����ǰ����
library(mice)
md.pattern(taiwan)  #����ȱʧ����,��������û��ȱʧֵ
set.seed(1234)      #����������ӣ������ظ�ʵ����
taiwan$default.payment.next.month=as.factor(taiwan$default.payment.next.month)  #���Ƿ�ΥԼת��Ϊ�����ͱ���
train=sample(1:nrow(taiwan), round(0.8* nrow(taiwan)))  #��ȡ80%,24000��������Ϊѵ����

######Bagging���ɷ��������þ�������Ϊ��������
library(randomForest)
taiwan_bag=randomForest(default.payment.next.month~., taiwan[train, ], mtry=round(sqrt(ncol(taiwan)),0)) 
#������Ļ��������ľ��������ñ���Ϊ���б�����p^1/2����  

###��Baggingģ�����ڲ��Լ�
taiwan_bag.pred = predict(taiwan_bag, taiwan[-train,-24])  #ȥ��ѵ�������ݣ���ͬʱȥ��Ԥ�����
accuracy=mean(taiwan_bag.pred==taiwan[-train,24]) * 100    #����׼ȷ��,��ȷ�ʴﵽ79.25%
print(sprintf('The bagging accuracy is %f', accuracy))

######���ɭ��
###�������ɭ��ģ�͡������Ҳ����bagging+������
taiwan_rf=randomForest(default.payment.next.month~., taiwan[train,],importance=T) #��ʾ������Ҫ��
###����
taiwan_rf.pred = predict(taiwan_rf, taiwan[-train,])        #ѡȡ�����������������Ŀ�������Ĭ�ϵ�
accuracy=mean(taiwan_rf.pred==taiwan[-train,24]) * 100      #����׼ȷ��,��ȷ�ʴﵽ80.25%
print(sprintf('The randomforest accuracy is %f', accuracy)) #���bagging���ƣ�ֻ����������Ŀ�����ʹ��׼ȷ���������


######Boosting,��gbm��ͨ����������������������
###����Boostingģ��
library(gbm)            #����gbm��
#�����ݶ���������Gradient Boosting
taiwan_boost=gbm(as.numeric(as.character(default.payment.next.month))~., taiwan[train,], distribution='bernoulli', 
               n.trees=5000, interaction.depth=4)         
#����������Ϊ���������γɵĻ�������Ϊ5000����Ҳ���ǵ�������Ϊ5000��
#ʹ���������Ե�ķֲ���Ҳ������ʧ����Ϊ��Ŭ���ֲ�����������һ��ѡ��bernoulli�ֲ�
#ÿ�������������Ϊ4��������
#ע����ֱ�������Ϊ0,1,��Ϊ��ֵ��
taiwan_boost.pred = predict(taiwan_boost, taiwan[-train,-24], n.trees=5000)
#������������
confusion(as.numeric(as.character(taiwan[-train,]$default.payment.next.month)) > 0, taiwan_boost.pred > 0)
#�����������ʾ�����Դ�����Ϊ23.84%�����Լ�����ȷ��Ϊ76.16%


#�����Գ�����adaboost��ͬ��������gbm������������Ӧ��������AdaBoost

#������Ҫ����xgboost���İ�װ��ֻ����R������install.packages(��xgboost��)���ɡ���װ�ð������Ƕ����ٰ����ݽ��н�ģԤ�⡣
library('xgboost')
x<-taiwan[train,1:23]
y<-taiwan[train,24]
x<-apply(x,2,as.numeric)               #��x���б���ת��Ϊ������
y<-as.numeric(y)-1              #��yҲת��Ϊ�����ͱ���
bst<-xgboost(data=x,label=y,max.depth=2,eta=1,nround=10,objective='binary:logistic')
#�ڴ˴�������ѡȡ������Ϊ2��ѧϰ����Ϊ1������������Ϊ10��Ŀ�꺯���ǻ��ڶ����������logistic��ʧ����ģ�ͽ�����
# xgboostѵ������
#Ҳ�������ú���xgb.cv�����ݽ��н�����֤���������ý�����֤ȷ���������,���ý������ȷ����ѵ�������
cv.res<-xgb.cv(data=x,label=y,max.depth=2,eta=1,nround=10,objective='binary:logistic',nfold=5)

#����predict��������Ԥ��
test=taiwan[-train,1:23]
test<-apply(test,2,as.numeric) 
pred<-predict(bst,test)
pred<-round(pred, 0)  #ת��Ϊ�������
table(as.factor(pred),as.factor(taiwan[-train,24]))	 #������������
accuracy=mean(as.factor(pred)==taiwan[-train,24]) * 100  #Ԥ��׼ȷ��Ϊ81.75%,�ٶȺܿ�
accuracy


# ���ذ�������
library(gbm)
data(PimaIndiansDiabetes2,package='mlbench')
# ����Ӧ����תΪ0-1��ʽ
data <- PimaIndiansDiabetes2
data$diabetes <- as.numeric(data$diabetes)
data <- transform(data,diabetes=diabetes-1)
# ʹ��gbm������ģ
model <- gbm(diabetes~.,data=data,shrinkage=0.01,
             distribution='bernoulli',cv.folds=5,
             n.trees=3000,verbose=F)
# �ý������ȷ����ѵ�������
best.iter <- gbm.perf(model,method='cv')