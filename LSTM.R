####################### Reverse Transformation Function  ###################################

## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


library(imputeTS)
library(readr)
library(plyr)
library(tidyverse)
library(fpp2)
library(forecast)
library(tseries)
library(ggplot2)
library(keras)
library(tensorflow)
library(sparklyr)
library(kerasR)
library(openxlsx)

####################################  Loading Data  #############################################



setwd("/Users/Desktop/Final CACC Data")

data = read_csv("STYLE009_BLUE_------.csv")

#data = read_csv("STYLE009_PINK_------.csv")

#data = read_csv("STYLE031_BLACK_407Q88.csv")

#data = read_csv("STYLE042_GREY_549310.csv")

#data = read_csv("STYLE082_WHITE_------.csv")

#data = read_csv("STYLE087_GREY_------.csv")

#data = read_csv("STYLE104_BLUE_Y06085.csv")

#data = read_csv("STYLE105_PURPLE_Y06145.csv")

#data = read_csv("STYLE111_RED_Y06145.csv")

#data = read_csv("STYLE116_BLACK_549314.csv")

#data = read_csv("STYLE179_BLUE_------.csv")

#data = read_csv("STYLE189_BLACK_------.csv")

#data = read_csv("STYLE203_GREY_549314.csv")

#data = read_csv("STYLE207_BLUE_407D55.csv")

#data = read_csv("STYLE264_RED_------.csv")

#data = read_csv("STYLE276_BLACK_549333.csv")

#data = read_csv("STYLE329_BLACK_40GEAR.csv")

#data = read_csv("STYLE357_GREY_------.csv")

#data = read_csv("STYLE359_BLUE_------.csv")

#data = read_csv("STYLE366_BLACK_------.csv")




######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_BLUE_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_BLUE_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 
   
##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_BLUE_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]

 

## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_BLUE_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#Style009_Pink

data = read_csv("STYLE009_PINK_------.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_PINK_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_PINK_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_PINK_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE009_PINK_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE031_BLACK_407Q88

data = read_csv("STYLE031_BLACK_407Q88.csv")

######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE031_BLACK_407Q88", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE031_BLACK_407Q88", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE031_BLACK_407Q88", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE031_BLACK_407Q88", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)





##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE042_GREY_549310

data = read_csv("STYLE042_GREY_549310.csv")

######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE042_GREY_549310", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE042_GREY_549310", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE042_GREY_549310", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE042_GREY_549310", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)








##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE082_WHITE_------

data = read_csv("STYLE082_WHITE_------.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE082_WHITE_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE082_WHITE_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE082_WHITE_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE082_WHITE_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)







##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE087_GREY_------

data = read_csv("STYLE087_GREY_------.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE087_GREY_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE087_GREY_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE087_GREY_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE087_GREY_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)







##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE104_BLUE_Y06085


data = read_csv("STYLE104_BLUE_Y06085.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE104_BLUE_Y06085", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE104_BLUE_Y06085", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE104_BLUE_Y06085", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE104_BLUE_Y06085", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)





##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE105_PURPLE_Y06145

data = read_csv("STYLE105_PURPLE_Y06145.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE105_PURPLE_Y06145", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE105_PURPLE_Y06145", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE105_PURPLE_Y06145", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}


############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE105_PURPLE_Y06145", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)




##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE111_RED_Y06145


data = read_csv("STYLE111_RED_Y06145.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:22], frequency = 26, start=c(1,1), end=c(1,22))
brick_test_real = ts(brick_data_ts[22:34], frequency = 26, start=c(1,23))

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(1,23)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE111_RED_Y06145", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,8)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE111_RED_Y06145", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:22], frequency = 26, start=c(1,1), end=c(1,23))
online_test_real = ts(online_data_ts[22:34], frequency = 26, start=c(1,24), end=c(2,10)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(1,22)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE111_RED_Y06145", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,10)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE111_RED_Y06145", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE116_BLACK_549314

data = read_csv("STYLE116_BLACK_549314.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE116_BLACK_549314", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE116_BLACK_549314", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE116_BLACK_549314", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE116_BLACK_549314", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE179_BLUE_------

data = read_csv("STYLE179_BLUE_------.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE179_BLUE_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE179_BLUE_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE179_BLUE_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE179_BLUE_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)




##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE189_BLACK_------

data = read_csv("STYLE189_BLACK_------.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE189_BLACK_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE189_BLACK_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE189_BLACK_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE189_BLACK_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)





##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE203_GREY_549314

data = read_csv("STYLE203_GREY_549314.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE203_GREY_549314", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE203_GREY_549314", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE203_GREY_549314", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE203_GREY_549314", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE207_BLUE_407D55

data = read_csv("STYLE207_BLUE_407D55.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE207_BLUE_407D55", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE207_BLUE_407D55", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE207_BLUE_407D55", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE207_BLUE_407D55", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)




##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE264_RED_------

data = read_csv("STYLE264_RED_------.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE264_RED_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE264_RED_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE264_RED_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE264_RED_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE276_BLACK_549333


data = read_csv("STYLE276_BLACK_549333.csv")


######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE276_BLACK_549333", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE276_BLACK_549333", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE276_BLACK_549333", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE276_BLACK_549333", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)






##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE329_BLACK_40GEAR

data = read_csv("STYLE329_BLACK_40GEAR.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE329_BLACK_40GEAR", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE329_BLACK_40GEAR", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE329_BLACK_40GEAR", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE329_BLACK_40GEAR", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)




##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE357_GREY_------

data = read_csv("STYLE357_GREY_------.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE357_GREY_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE357_GREY_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE357_GREY_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE357_GREY_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE359_BLUE_------

data = read_csv("STYLE359_BLUE_------.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE359_BLUE_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE359_BLUE_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE359_BLUE_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE359_BLUE_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################
##############################################################################################################

#STYLE366_BLACK_------

data = read_csv("STYLE366_BLACK_------.csv")



######################  Preparing Data for Time Series Modeling #################################


## Splitting Data by CHANNEL
brick_data <- subset(data, CHANNEL == "BrickandMortar")
online_data <- subset(data, CHANNEL == "Online")


## Dropping Unneeded Columns
brick_data <- within(brick_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))
online_data <- within(online_data, rm(CHANNEL, FISCAL_YEAR, FISCAL_WEEK))


##############################  Imputting Missing Values  ######################################


## Imputting Missing Data
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
imputted_brick <- na.seadec(brick_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
imputted_online <- na.seadec(online_data, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation

##############################  Create Time Series Objects  ######################################


## Converting DataFrames into Time Series Type
#brick_data_ts = ts(imputted_brick[1:52,1], frequency = 26) 
brick_data_ts = ts(imputted_brick, frequency = 26) 
#online_data_ts = ts(imputted_online[1:52,1], frequency = 26) 
online_data_ts = ts(imputted_online, frequency = 26) 

## Plotting Data
autoplot(brick_data_ts)
autoplot(online_data_ts)




#########################  Checking if Data is Stationary #################################



## Checking if Data is Stationary (Currently not stationary but working on improving that)

ggAcf(brick_data_ts) 
ggAcf(online_data_ts)

adf.test(brick_data_ts, alternative = "stationary")
adf.test(online_data_ts, alternative = "stationary")

diff_brick <- diff(brick_data_ts, differences=1) # Differencing drops one row
diff_online <- diff(online_data_ts, differences=1)

adf.test(diff_brick, alternative = "stationary")
adf.test(diff_online, alternative = "stationary")

#best_lambda_g <- BoxCox.lambda(brick) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#brick <- brick %>% BoxCox(lambda = best_lambda_g)

#best_lambda_g <- BoxCox.lambda(online) # Ensuring Data Normality by Finding the Best Lambda Value with Geurrero Method
#online <- online %>% BoxCox(lambda = best_lambda_g)

ggAcf(diff_brick) 
ggAcf(diff_online)






######################## Convert Data into a Supervised Learning Dataset  ##############################


lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised_brick = lag_transform(diff_brick, 1)
supervised_online = lag_transform(diff_online, 1)

head(supervised_brick)
head(supervised_online)

############################ Splitting Data ###################################

N = nrow(supervised_brick)
n = round(N *0.75, digits = 0)
brick_train = supervised_brick[1:n, ]
brick_test  = supervised_brick[(n+1):N,  ]

N = nrow(supervised_online)
n = round(N *0.75, digits = 0)
online_train = supervised_online[1:n, ]
online_test  = supervised_online[(n+1):N,  ]


########################### Scale Data Min/Max  #####################################

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  y = test
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((y - min(y) ) / (max(y) - min(y)  ))
  
  scaled_train = std_train *(fr_max - fr_min) + fr_min
  scaled_test = std_test *(fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler_train= c(min =min(x), max = max(x), scaler_test= c(min =min(y), max = max(y)))))
  
}


Scaled_brick = scale_data(brick_train, brick_test, c(-1, 1))

y_train_brick = Scaled_brick$scaled_train[, 2]
x_train_brick = Scaled_brick$scaled_train[, 1]

y_test_brick = Scaled_brick$scaled_test[, 2]
x_test_brick = Scaled_brick$scaled_test[, 1]


Scaled_online = scale_data(online_train, online_test, c(-1, 1))

y_train_online = Scaled_online$scaled_train[, 2]
x_train_online = Scaled_online$scaled_train[, 1]

y_test_online = Scaled_online$scaled_test[, 2]
x_test_online = Scaled_online$scaled_test[, 1]



######################## Brick Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 42) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_brick) <- c(length(x_train_brick), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_brick)[2]
X_shape3 = dim(x_train_brick)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                           # can adjust this, in model tuninig phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick, y_train_brick, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


################################### Brick Model Validation  ###########################################

L = length(x_test_brick)
scaler = Scaled_brick$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_brick[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



brick_train_real = ts(brick_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
brick_test_real = ts(brick_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

############################################################################################################################

brick_train_real %>% autoplot(main = "Brick and Mortar - LSTM Train_Test", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(2,13)), size =2, series = "Prediction") +
  autolayer(brick_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE366_BLACK_------", 
         filename = "Brick and Mortar - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(brick_test_real, predictions)

autoplot(brick_data_ts)




#################################################### Brick Forecast ###########################################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_brick_all = scale_data_all(supervised_brick, c(-1, 1))

y_train_brick_all = Scaled_brick_all$scaled_data[, 2]
x_train_brick_all = Scaled_brick_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_brick_all) <- c(length(x_train_brick_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_brick_all)[2]
X_shape3 = dim(x_train_brick_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_brick_all, y_train_brick_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_brick_all$scaler
predictions = numeric(L)


for(i in 1:L){
  X = x_train_brick_all[i]              # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + brick_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

##############################################################################################################################

brick_data_ts %>% autoplot(main = "Brick and Mortar - LSTM Forecast", ylab = "Demand", series = "Train") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE366_BLACK_------", 
         filename = "Brick and Mortar - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(brick_data_ts)





######################## Online Train / Test LSTM Model Creation ####################################

kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104) # https://stackoverflow.com/questions/44591138/making-neural-network-training-reproducible-using-rstudios-keras-interface

# Reshape the input to 3-dim
dim(x_train_online) <- c(length(x_train_online), 1, 1)

# specify required arguments
X_shape2 = dim(x_train_online)[2]
X_shape3 = dim(x_train_online)[3]
batch_size = 1                      # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuninig phase


model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online, y_train_online, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



################################### Online Model Validation  ###########################################


L = length(x_test_online)
scaler = Scaled_online$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test_online[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}



online_train_real = ts(online_data_ts[1:38], frequency = 26, start=c(1,1), end=c(2,12))
online_test_real = ts(online_data_ts[39:51], frequency = 26, start=c(2,13), end=c(2,25)) 

##################################################################################################################################

online_train_real %>% autoplot(main = "Online - LSTM Train_Test", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start = c(2,13)),size = 2, series = "Prediction") +
  autolayer(online_test_real, series="Test", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE366_BLACK_------", 
         filename = "Online - LSTM Train_Test.png", width=9, height=6,dpi=500)


accuracy(online_test_real, predictions)

autoplot(online_data_ts)


################################### Online Forecast ###################################

# Min / Max Scaling

scale_data_all = function(train, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))
  
  scaled_data = std_data *(fr_max - fr_min) + fr_min
  
  return( list(scaled_data = as.vector(scaled_data),scaler_data= c(min =min(x), max = max(x))))
  
}



Scaled_online_all = scale_data_all(supervised_online, c(-1, 1))

y_train_online_all = Scaled_online_all$scaled_data[, 2]
x_train_online_all = Scaled_online_all$scaled_data[, 1]



## Model Creation


# Reshape the input to 3-dim
dim(x_train_online_all) <- c(length(x_train_online_all), 1, 1) # (batch_size, timesteps, input_dim)

# specify required arguments
X_shape2 = dim(x_train_online_all)[2]
X_shape3 = dim(x_train_online_all)[3]
batch_size = 1                       # must be a common factor of both the train and test samples
units = 1                            # can adjust this, in model tuning phase




model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)

summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train_online_all, y_train_online_all, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


## Forecast


L = length(1:13)
scaler = Scaled_online_all$scaler
predictions = numeric(L)

#empty_vector <- vector(mode="numeric", length=13)


for(i in 1:L){
  X = x_train_online_all[i]           # Not Sure if this is the correct way to do this
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + online_data_ts[(n+i)]
  # store
  predictions[i] <- yhat
}

############################################################################################################################################

online_data_ts %>% autoplot(main = "Online - LSTM Forecast", ylab = "Demand") + 
  autolayer(ts(predictions, frequency = 26, start=c(3,1)), series = "Prediction", size = 2)+
  geom_line(size =1) +
  theme(plot.title = element_text(size =22), axis.text = element_text(size=18),
        axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
  ggsave(path = "C:/Users/Michael Wile/Desktop/Final CACC Graphs/STYLE366_BLACK_------", 
         filename = "Online - LSTM Forecast.png", width=9, height=6,dpi=500)

autoplot(online_data_ts)



