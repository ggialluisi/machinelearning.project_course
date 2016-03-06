
install.packages("ggplot2")

install.packages("caret")
require(caret)

install.packages('e1071', dependencies=TRUE)
install.packages('RANN', dependencies=TRUE)

install.packages("AppliedPredictiveModeling")
require(AppliedPredictiveModeling)


install.packages("kernlab")
library(kernlab)
data(spam)


inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

dim(training)
dim(testing)


#K-FOLD:
set.seed(32323)
folds <- createFolds(y=spam$type, k=10,
                     list=TRUE, #cada fold vem como uma lista ou, se false-> vetor
                     returnTrain=TRUE) #retorna o traning set ou, se false-> test set

sapply(folds,length)#pra ver lenght de cada fold...

folds[[2]][1:10]


#RESAMPLE:  (resampling ->> with replacement - pega sample aleatorio, amostras podem repetir
set.seed(32323)
folds <- createResample(y=spam$type, times=10,
                     list=TRUE) #cada fold vem como uma lista ou, se false-> vetor

sapply(folds,length)#pra ver lenght de cada fold...

folds[[2]][1:10]


#TIME SLICES
set.seed(32323)
tme <- 1:1000 #tamanho de uma unidade de tempo
folds <- createTimeSlices(y=tme, 
                          initialWindow = 20, #janela com 20 * tme
                          horizon=10) #quantas tme quer prever

names(folds)#pra ver lenght de cada fold...
folds$train[1]
folds$test[1]

folds$train[2]
folds$test[2]

length(folds$train)







data(spam)


inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]


# opçoes para TRINAMENTO:

args(trainControl)



#plot predictors

install.packages("ISLR")
require(ISLR)
data(Wage)

inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

#kind of pairs
featurePlot(x=training[,c("age", "education", "jobclass")],
            y=training$wage,
            plot="pairs")

#qplot
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)

#add smooths:
qq <- qplot(age,wage,colour=education,data=training)
qq + geom_smooth(method='lm', formua=y~x)


#cut2: making factors
install.packages("Hmisc")
require(Hmisc)

install.packages("gridExtra")
require(gridExtra)

cutWage <- cut2(training$wage,g=3)
table(cutWage)
p1 <- qplot(cutWage, age, data=training,
            fill=cutWage,
            geom=c("boxplot"))
p2 <- qplot(cutWage, age, data=training,
            fill=cutWage,
            geom=c("boxplot", "jitter"))
grid.arrange(p1,p2,ncol=2)

#tabela...
table(cutWage, training$jobclass)
prop.table(table(cutWage, training$jobclass), 1)#por linha
prop.table(table(cutWage, training$jobclass), 2)#por coluna

#density...
qplot(wage, colour=education, data=training, geom="density")




#preprocessing




inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

dim(training)
dim(testing)

hist(training$capitalAve, main="", xlab = "ave. capital run lenght")

trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)

mean(trainCapAveS)
sd(trainCapAveS)

testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve))/sd(trainCapAve) # mean e sd do TREIN mesmo, precisa aplicar a mesma transformaçao!
mean(testCapAveS)
sd(testCapAveS)



preObj <- preProcess(training[,-58], method=c("center", "scale"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)


testCapAveS <- predict(preObj,testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)


# passar opções de preProcess para metodo train como parametro

set.seed(32343)
modelFit <- train(type~.,data=training,
                  preProcess=c("center","scale"), method="glm")
modelFit


# BoxCox: tenta normalizar
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)


# IMPUT DATA para missing data
set.seed(13343)

# cria coluna capAve e coloca alguns NAs nela
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1,prob=0.05)==1
training$capAve[selectNA] <- NA

# IMPUT valores usando knnImpute
preObj <- preProcess(training[,-58],method="knnImpute") # acha k nearest parecidos e faz a média do valor que falta
capAve <- predict(preObj,training[,-58])$capAve


capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)

# verifica que
quantile(capAve - capAveTruth)

quantile((capAve - capAveTruth)[selectNA])

quantile((capAve - capAveTruth)[!selectNA])



# Covariate CREATION - criaçao de FEATURES (=Predictors)

inTrain <- createDataPartition(y=Wage$wage,
                               p=.7, list=FALSE)
training <- Wage[inTrain,];testing <- Wage[-inTrain,];



# cria colunas a partir de FACTORS! 
table(training$jobclass)
dummies <- dummyVars(wage~jobclass,data=training)
head(predict(dummies,newdata=training))

# REMOVER features com ZERO COVARIATE
nsv <-nearZeroVar(training,saveMetrics = TRUE)
nsv
# neste exemplo, sex e region apresenta 
# variaçao perto de zero, não devem ser usadas 
# como features...


#google:
# feature extraction for [data type] (email, image, people)

library(splines)
bsBasis 



# PRINCIPAL COMPONENTS ANALYSIS (PCA)- reduz variáveis para 2, com operaçao otimizadas de todas variáveis

modelFit <- train(training$type ~ ., method="glm",preProcess = "pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing))








library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)


adData = data.frame(predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

dim(training); dim(testing); 

diagnosis,
names(predictors)
str(diagnosis)




library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]


training$Superplasticizer[training$Superplasticizer==0]
hist(training$Superplasticizer)
sum(training$Superplasticizer==0)
log(0)


library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors[,57:68])
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


modelFit_noPCA <- train(diagnosis ~ ., method="glm", data=training)
confusionMatrix(testing$diagnosis, predict(modelFit_noPCA, testing))




preObj <- preProcess(training[,-1], method=c("pca"), thresh = .80)
trainPCA <- predict(preObj,training[,-1])
trainPCA <- data.frame(training$diagnosis,trainPCA)
names(trainPCA)[1] <- "diagnosis"

testPCA <- predict(preObj,testing[,-1])
testPCA <- data.frame(testing$diagnosis,testPCA)
names(testPCA)[1] <- "diagnosis"

modelFit_yesPCA <- train(diagnosis ~ ., method="glm", data=trainPCA)
confusionMatrix(testPCA$diagnosis, predict(modelFit_yesPCA, testPCA))





str(testPCA)

str(trainPCA)


modelFit_yesPCA <- train(training$diagnosis ~ ., method="glm",preProcess = "pca", data=training)


names(training)
head(training[,58:69])


preObj <- preProcess(training[,58:69], method=c("pca"), thresh = .80)
trainPCA <- predict(preObj,training[,58:69])




head(trainPCA)

modelFit <- train(training$type ~ ., method="glm",preProcess = "pca", data=training)

confusionMatrix(testing$type, predict(modelFit, testing))





library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]



# WEEK 3

# TREES

data(iris)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species, 
                               p=.7, list=FALSE)
training<-iris[inTrain,];testing<-iris[-inTrain,]

qplot(Petal.Width, Sepal.Width, colour=Species, data=training)

modFit <- train(Species~.,method ="rpart", data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE,
     main="Classification Tree")
text(modFit$finalModel, use.n = TRUE, all=TRUE, cex=.8)


install.packages("rpart.plot")
require(rpart.plot)
install.packages("rattle")
require(rattle)
fancyRpartPlot(modFit$finalModel)



# BAGGING

#  resample depois tira média dos resultados

install.packages("ElemStatLearn")
library(ElemStatLearn)
data(ozone,package="ElemStatLearn")
ozone<-ozone[order(ozone$ozone),]
head(ozone)

# prever temperatura dado ozone

ll <- matrix(NA, nrow=10,ncol=155)
for(i in 1:10){
    ss <- sample(1:dim(ozone)[1],replace=T)
    ozone0 <- ozone[ss,]
    ozone0 <- ozone0[order(ozone0$ozone),]

    loess0 <- loess(temperature ~ ozone, data=ozone0, span=0.2)
    ll[i,] <- predict(loess0,newdata=data.frame(ozone=1:155))
}

plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){
    lines(1:155, ll[i,],col="gray",lwd=2)
}
lines(1:155, apply(ll,2,mean),col="red",lwd=2)



# RANDOM FOREST

# bootsrap variaveis também!

data(iris); library(ggplot2)
inTrain <- createDataPartition(y=iris$Species,
                               p=.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species~.,data=training,
                method="rf",prox=TRUE)
modFit


pred <- predict(modFit, testing)
testing$predRight <- pred==testing$Species
table(pred,testing$Species)
qplot(Petal.Width,Petal.Length,colour=predRight,data=testing,main="nedData Predictions")



# BOOSTING

library(ISLR)
data(Wage)

Wage <- subset(Wage,select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]


modFit <- train(wage~.,
                method="gbm", #gbm: boosting com trees
                data=training, verbose=FALSE)

print(modFit)

qplot(predict(modFit,testing), wage, data=testing)


# MODEL BASED PREDICTION
data(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)
dim(testing)

modlda <- train(Species~.,data=training, method="lda") 
modnb <- train(Species~.,data=training, method="nb") 

plda <- predict(modlda,testing)
pnb <- predict(modnb,testing)
table(plda, pnb)

equalPredictions <- (plda==pnb)
qplot(Petal.Width,Sepal.Width,colour=equalPredictions,data=testing)





## QUIZ 3
install.packages("pgmm")
library(pgmm)
data(olive)
head(olive)
olive = olive[,-1]










# medindo CORRELAÇÃO ENTRE VARIÁVEIS
# vídeo da semana week 2 - Preprocessing with principal component analysis

require(caret)
require(kernlab)
data(spam)
inTrain = createDataPartition(y=spam$type, p=0.75, list = FALSE)
training = spam[inTrain,]
testing = spam[-inTrain,]


##!!
M <- abs(cor(training[,-58])) #->> calcula correlaçao entre todos (menos a coluna 58, que é a coluna type)
diag(M) <- 0     #>> diagonal é a correlaçao da variavel com ela mesma, valor 1, não nos interessa, zera a coluna para não aparecer no which abaixo
which(M > 0.8, arr.ind = T) #mostra quais as colunas tem valores maiores que 0.8

plot(spam[,34], spam[,32])


# lista 15 maiores correlações com a variável
M2 <- abs(cor(training[,-58], training[,58]=="spam"))
head(M2[order(-M2),], n = 15)
which(M2 > 0.2181,arr.ind = T)
head(training[, which(M2 > 0.2181,arr.ind = F)])










# QUIZ 2

# 1)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

head(training)
dim(training)
dim(testing)


adData = data.frame(predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]


# 2)

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]



names <- colnames(concrete)
names <- names[-length(names)]
names(mixtures)
head(mixtures)

featurePlot(x = training[, names], y = training$CompressiveStrength, plot = "pairs")

index <- seq_along(1:nrow(training))
ggplot(data = training, aes(x = index, y = CompressiveStrength)) + geom_point() + 
    theme_bw()

library(Hmisc)
cutCS <- cut2(training$CompressiveStrength, g = 4)
summary(cutCS)

ggplot(data = training, aes(y = index, x = cutCS)) + geom_boxplot() + geom_jitter(col = "blue") + 
    theme_bw()

featurePlot(x = training[, names], y = cutCS, plot = "box")


# 4)

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


IL_str <- grep("^IL", colnames(training), value = TRUE)
preProc <- preProcess(training[, IL_str], method = "pca", thresh = 0.9)
preProc$rotation



# Question 5

# Load the Alzheimer's disease data using the commands:


library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


# Create a training data set consisting of only the predictors 
# with variable names beginning with IL and the diagnosis. 
# Build two predictive models, one using the predictors as they are 
# and one using PCA with principal components explaining 80% of the 
# variance in the predictors. Use method="glm" in the train function. 
#What is the accuracy of each method in the test set? Which is more accurate?


set.seed(3433)
## grep the predictors starting with 'IL'
IL_str <- grep("^IL", colnames(training), value = TRUE)
## make a subset of these predictors
predictors_IL <- predictors[, IL_str]
df <- data.frame(diagnosis, predictors_IL)
inTrain = createDataPartition(df$diagnosis, p = 3/4)[[1]]
training = df[inTrain, ]
testing = df[-inTrain, ]

## train the data using the first method
modelFit <- train(diagnosis ~ ., method = "glm", data = training)



predictions <- predict(modelFit, newdata = testing)
## get the confustion matrix for the first method
C1 <- confusionMatrix(predictions, testing$diagnosis)
print(C1)



A1 <- C1$overall[1]
## usando PCA:
## do similar steps with the caret package
modelFit <- train(training$diagnosis ~ ., method = "glm", preProcess = "pca", 
                  data = training, trControl = trainControl(preProcOptions = list(thresh = 0.8)))
C2 <- confusionMatrix(testing$diagnosis, predict(modelFit, testing))
print(C2)


A2 <- C2$overall[1]


# A1 = 0.65
# A2 = 0.72






