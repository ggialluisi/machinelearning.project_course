# Prediction Assignment Writeup
### Johns Hopkins University
#### Practical Machine Learning Course Project
#### Gustavo P. Gialluisi
##### 05/03/2016


###Overview

This report is a basic Machine Learning excercise that intend to analyse data in the Weight Lifting Exercise Dataset to predict values of column _classe_ of the dataset.  

A good model was fit after removing columns that had high percentage of NA values. Columns with Near Zero Variance were also removed. 

A accuracy of 0.99 was obtained using RandomForest model.


### Analysis Setup

Load required R packages. And load the data. And set seed for reproducible results.


```r
### Load required R packages
require(caret)
require(ggplot2)

### Load data
trainsubset <- read.csv("pml-training.csv",na.strings = c("NA", "#DIV/0!"))
tosubmit <- read.csv("pml-testing.csv",na.strings = c("NA", "#DIV/0!"))

### Set seed for reproducible results
set.seed(123456)
```


Next I'll create training and testing partitions, so I can use only the training partition to choose the features.


```r
inTrain <- createDataPartition(y=trainsubset$classe,
                               p=0.7, list=FALSE)
training <- trainsubset[inTrain,]
testing <- trainsubset[-inTrain,]
```


### Choosing Columns

Let's have a look at the dataset structure.


```r
# 'testy' is my test dataset, that is a copy of 'training' dataset
testy <- training 
dim(testy)
```

```
## [1] 13737   160
```

The dataset has 160 variables.  


```r
str(testy[,1:15])
```

```
## 'data.frame':	13737 obs. of  15 variables:
##  $ X                   : int  1 3 5 7 8 10 11 13 14 16 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 820366 196328 368296 440390 484434 500302 560359 576390 644302 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.42 1.48 1.42 1.42 1.45 1.45 1.42 1.42 1.48 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.09 8.13 8.17 8.18 8.2 8.21 8.15 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt  : num  NA NA NA NA NA NA NA NA NA NA ...
```

'X' column is only an index, not a predictor, must remove it.  
And since I'm not doing a time series analysis, I'll remove the timestamp variable 'cvtd_timestamp'.  


```r
testy <- subset(testy, select = -c(X))
testy <- subset(testy, select = -c(cvtd_timestamp))
```

#### Near Zero Variance Analysis
The following variables has Near Zero Variance:


```r
# near zero variance analysis
nzv <- nearZeroVar(testy, saveMetrics = TRUE)

# columns to remove:
row.names(nzv[nzv$nzv,])
```

```
##  [1] "new_window"             "kurtosis_yaw_belt"     
##  [3] "skewness_yaw_belt"      "amplitude_yaw_belt"    
##  [5] "avg_roll_arm"           "stddev_roll_arm"       
##  [7] "var_roll_arm"           "avg_pitch_arm"         
##  [9] "stddev_pitch_arm"       "var_pitch_arm"         
## [11] "avg_yaw_arm"            "stddev_yaw_arm"        
## [13] "var_yaw_arm"            "kurtosis_yaw_dumbbell" 
## [15] "skewness_yaw_dumbbell"  "amplitude_yaw_dumbbell"
## [17] "kurtosis_yaw_forearm"   "skewness_yaw_forearm"  
## [19] "max_roll_forearm"       "min_roll_forearm"      
## [21] "amplitude_roll_forearm" "amplitude_yaw_forearm" 
## [23] "avg_roll_forearm"       "stddev_roll_forearm"   
## [25] "var_roll_forearm"       "avg_pitch_forearm"     
## [27] "stddev_pitch_forearm"   "var_pitch_forearm"     
## [29] "avg_yaw_forearm"        "stddev_yaw_forearm"    
## [31] "var_yaw_forearm"
```


So, this command remove the Near Zero Variance variables from my subset:


```r
testy <- subset(testy, select = -c(new_window, kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, max_roll_arm, amplitude_roll_arm, amplitude_pitch_arm, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm, max_roll_forearm, min_roll_forearm, amplitude_roll_forearm, amplitude_yaw_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm))
```

#### Too much 'NA's

Many variables have a high percentage of 'NA' values.


```r
#create a dataset with the average of 'NA' values of each variable
na.analysis <- as.data.frame(colMeans(is.na(testy)))
names(na.analysis) <- c("na.pc")
na.analysis$variable <- rownames(na.analysis)

#round values to 2 digits
na.analysis$na.pc <- round(na.analysis$na.pc, 2)

#create a table
table(na.analysis$na.pc)
```

```
## 
##    0 0.98 
##   57   67
```


This table view above shows that we have 57 variables with not even one 'NA' value, and all other 67 remaining variables have more than 98% of 'NA' values.  

So this variables with more than 0% of 'NA' values will be removed:


```r
na.analysis[na.analysis$na.pc > 0,]$variable
```

```
##  [1] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##  [3] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [5] "max_roll_belt"            "max_picth_belt"          
##  [7] "max_yaw_belt"             "min_roll_belt"           
##  [9] "min_pitch_belt"           "min_yaw_belt"            
## [11] "amplitude_roll_belt"      "amplitude_pitch_belt"    
## [13] "var_total_accel_belt"     "avg_roll_belt"           
## [15] "stddev_roll_belt"         "var_roll_belt"           
## [17] "avg_pitch_belt"           "stddev_pitch_belt"       
## [19] "var_pitch_belt"           "avg_yaw_belt"            
## [21] "stddev_yaw_belt"          "var_yaw_belt"            
## [23] "var_accel_arm"            "kurtosis_roll_arm"       
## [25] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
## [27] "skewness_roll_arm"        "skewness_pitch_arm"      
## [29] "skewness_yaw_arm"         "max_picth_arm"           
## [31] "max_yaw_arm"              "min_roll_arm"            
## [33] "min_pitch_arm"            "min_yaw_arm"             
## [35] "amplitude_yaw_arm"        "kurtosis_roll_dumbbell"  
## [37] "kurtosis_picth_dumbbell"  "skewness_roll_dumbbell"  
## [39] "skewness_pitch_dumbbell"  "max_roll_dumbbell"       
## [41] "max_picth_dumbbell"       "max_yaw_dumbbell"        
## [43] "min_roll_dumbbell"        "min_pitch_dumbbell"      
## [45] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
## [47] "amplitude_pitch_dumbbell" "var_accel_dumbbell"      
## [49] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
## [51] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
## [53] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
## [55] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
## [57] "var_yaw_dumbbell"         "kurtosis_roll_forearm"   
## [59] "kurtosis_picth_forearm"   "skewness_roll_forearm"   
## [61] "skewness_pitch_forearm"   "max_picth_forearm"       
## [63] "max_yaw_forearm"          "min_pitch_forearm"       
## [65] "min_yaw_forearm"          "amplitude_pitch_forearm" 
## [67] "var_accel_forearm"
```


Command to remove them from the dataset:


```r
    testy <- subset(testy, select = -c(kurtosis_roll_belt, kurtosis_picth_belt, skewness_roll_belt, skewness_roll_belt.1, max_roll_belt, max_picth_belt, max_yaw_belt, min_roll_belt, min_pitch_belt, min_yaw_belt, amplitude_roll_belt, amplitude_pitch_belt, var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt, var_pitch_belt, avg_yaw_belt, stddev_yaw_belt, var_yaw_belt, var_accel_arm, kurtosis_roll_arm, kurtosis_picth_arm, kurtosis_yaw_arm, skewness_roll_arm, skewness_pitch_arm, skewness_yaw_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_yaw_arm, kurtosis_roll_dumbbell, kurtosis_picth_dumbbell, skewness_roll_dumbbell, skewness_pitch_dumbbell, max_roll_dumbbell, max_picth_dumbbell, max_yaw_dumbbell, min_roll_dumbbell, min_pitch_dumbbell, min_yaw_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell, stddev_yaw_dumbbell, var_yaw_dumbbell, kurtosis_roll_forearm, kurtosis_picth_forearm, skewness_roll_forearm, skewness_pitch_forearm, max_picth_forearm, max_yaw_forearm, min_pitch_forearm, min_yaw_forearm, amplitude_pitch_forearm, var_accel_forearm))
```


#### The 'transformIt' function

Now this command creates the function 'transformIt(dataset)' to remove all undesired columns:  


```r
transformIt <- function(dataset){

    # function returns the subset
    subset(dataset, select = -c(X, cvtd_timestamp, new_window, kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, max_roll_arm, amplitude_roll_arm, amplitude_pitch_arm, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm, max_roll_forearm, min_roll_forearm, amplitude_roll_forearm, amplitude_yaw_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm, kurtosis_roll_belt, kurtosis_picth_belt, skewness_roll_belt, skewness_roll_belt.1, max_roll_belt, max_picth_belt, max_yaw_belt, min_roll_belt, min_pitch_belt, min_yaw_belt, amplitude_roll_belt, amplitude_pitch_belt, var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt, var_pitch_belt, avg_yaw_belt, stddev_yaw_belt, var_yaw_belt, var_accel_arm, kurtosis_roll_arm, kurtosis_picth_arm, kurtosis_yaw_arm, skewness_roll_arm, skewness_pitch_arm, skewness_yaw_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_yaw_arm, kurtosis_roll_dumbbell, kurtosis_picth_dumbbell, skewness_roll_dumbbell, skewness_pitch_dumbbell, max_roll_dumbbell, max_picth_dumbbell, max_yaw_dumbbell, min_roll_dumbbell, min_pitch_dumbbell, min_yaw_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell, stddev_yaw_dumbbell, var_yaw_dumbbell, kurtosis_roll_forearm, kurtosis_picth_forearm, skewness_roll_forearm, skewness_pitch_forearm, max_picth_forearm, max_yaw_forearm, min_pitch_forearm, min_yaw_forearm, amplitude_pitch_forearm, var_accel_forearm))

    
}
```

I created a function so that I can apply it to all datasets:  


```r
training <- transformIt(training)
testing <- transformIt(testing)
tosubmit <- transformIt(tosubmit)
```


### Training the Model

I tried two algorithms, Random Forest and Gradient Boosting, to fit the model, and both presented high accuracy of more than 99%.  

I used 'center' and 'scale' pre-process methods because I understand that the fitting algorithms work better this way, with all numeric values centered and scaled between -1 and 1, and with all Standard Deviation scaled to 1.  

#### Random Forest Model

This command fit the training dataset to the Random Forest model:


```r
modelRF <- train(classe ~ ., data=training,
                  preProcess=c("center","scale"),
                  method="rf")
```

The command above took more than 10 hours to finish its execution... So I used the following command to save the resulting model to a rds file on my working directory:


```r
saveRDS(modelRF, file="modelRF.rds")
```

Then, when I need it again, just have to load it with readRDS command:


```r
modelRF <- readRDS(file="modelRF.rds")
```

Let's have a look in the Confusion Matrix, that compares results of prediction applying the resulting model to the 'testing' dataset. It shows an enormous accuracy of 99%:

```r
confusionMatrix(testing$classe, predict(modelRF, testing))
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-10
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    4 1135    0    0    0
##          C    0    2 1024    0    0
##          D    0    0    3  961    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9969, 0.9992)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9982   0.9971   0.9990   1.0000
## Specificity            1.0000   0.9992   0.9996   0.9994   0.9998
## Pos Pred Value         1.0000   0.9965   0.9981   0.9969   0.9991
## Neg Pred Value         0.9991   0.9996   0.9994   0.9998   1.0000
## Prevalence             0.2851   0.1932   0.1745   0.1635   0.1837
## Detection Rate         0.2845   0.1929   0.1740   0.1633   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9988   0.9987   0.9983   0.9992   0.9999
```



#### Gradient Boosting Model

Same for Gradient Boosting Model:


```r
model_gbm <- train(classe ~ ., data=training,
                  preProcess=c("center", "scale"), 
                  method="gbm", verbose=FALSE)

saveRDS(model_gbm, file="model_gbm.rds")
```



```r
model_gbm <- readRDS(file="model_gbm.rds")
confusionMatrix(testing$classe, predict(model_gbm, testing))
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    4 1131    4    0    0
##          C    0    2 1022    2    0
##          D    0    0    2  960    2
##          E    0    1    0    6 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9958          
##                  95% CI : (0.9937, 0.9972)
##     No Information Rate : 0.2848          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9956   0.9942   0.9917   0.9981
## Specificity            0.9995   0.9983   0.9992   0.9992   0.9985
## Pos Pred Value         0.9988   0.9930   0.9961   0.9959   0.9935
## Neg Pred Value         0.9991   0.9989   0.9988   0.9984   0.9996
## Prevalence             0.2848   0.1930   0.1747   0.1645   0.1830
## Detection Rate         0.2841   0.1922   0.1737   0.1631   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9986   0.9970   0.9967   0.9955   0.9983
```

### Conclusion

Both models got high accuracy. Using 'predict' function with RandomForest model and the 'tosubmit' dataset, we may get the answers for our Course Project Prediction Quizz:


```r
predict(modelRF, tosubmit)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

THANK YOU FOR REVIEWING THIS!

### My hardware configuration


```r
sessionInfo()  ## for reference
```

```
## R version 3.2.2 (2015-08-14)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 7 x64 (build 7601) Service Pack 1
## 
## locale:
## [1] LC_COLLATE=Portuguese_Brazil.1252  LC_CTYPE=Portuguese_Brazil.1252   
## [3] LC_MONETARY=Portuguese_Brazil.1252 LC_NUMERIC=C                      
## [5] LC_TIME=Portuguese_Brazil.1252    
## 
## attached base packages:
## [1] parallel  splines   stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
## [1] plyr_1.8.3          gbm_2.1.1           survival_2.38-3    
## [4] randomForest_4.6-10 caret_6.0-47        ggplot2_1.0.1      
## [7] lattice_0.20-33    
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.2         formatR_1.2.1       nloptr_1.0.4       
##  [4] class_7.3-13        iterators_1.0.7     tools_3.2.2        
##  [7] digest_0.6.9        lme4_1.1-8          evaluate_0.8       
## [10] nlme_3.1-121        gtable_0.1.2        mgcv_1.8-7         
## [13] Matrix_1.2-2        foreach_1.4.2       yaml_2.1.13        
## [16] brglm_0.5-9         SparseM_1.6         proto_0.3-10       
## [19] e1071_1.6-4         BradleyTerry2_1.0-6 stringr_1.0.0      
## [22] knitr_1.12          gtools_3.5.0        grid_3.2.2         
## [25] nnet_7.3-10         rmarkdown_0.3.3     minqa_1.2.4        
## [28] reshape2_1.4.1      car_2.0-25          magrittr_1.5       
## [31] scales_0.2.5        codetools_0.2-14    htmltools_0.2.6    
## [34] MASS_7.3-42         pbkrtest_0.4-2      colorspace_1.2-6   
## [37] quantreg_5.11       stringi_1.0-1       munsell_0.4.2
```
