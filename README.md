# Demand-Prediction-with-Neural-Network
Using LSTM to predict the success or failure of new products to be launched by Champion.
I successfully completed this project for HanesBrands at the National Champion Analytics Competition
*Note: Data used in this project will not be available for public view.

![Screen Shot 2020-08-02 at 7 57 28 PM](https://user-images.githubusercontent.com/47016027/89136027-8452f680-d4ff-11ea-84d8-1ecd535e252d.png)
# Introduction
The goal of this project is to predict a 13-week demand for Champion's new products and identify whether the products will succeed or fail online vs their Bricks & Mortar stores and how soon success or failure can be spotted in order to make plans accordingly.
# Methodology
We built three different models:
1.	Seasonal Naive: This method is like the naive method but predicts the last observed value of the same season of the year. This method works for highly seasonal data like my dataset.
2.	ARIMA: ARIMA stands for Autoregressive Integrated Moving Average model. It is a forecasting technique that projects the future values of a series based entirely on its own inertia. Its main application is in the area of short-term forecasting.
3.	Long short-term memory (LSTM): This is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.

![image](https://user-images.githubusercontent.com/47016027/89136176-591cd700-d500-11ea-9e5a-3302129a3458.png)
# Evaluation
LSTM model performed very well overall. Generally, we got an average RMSE of around 4 in our test/train set. Some were higher and some were lower than that. LSTM generally improved prediction by more than double, on average, in comparison to the ARIMA model. ARIMA performed alright with some products but there was a lot of room for improvement

![image](https://user-images.githubusercontent.com/47016027/89136312-01cb3680-d501-11ea-9857-4773a57e13ba.png)
# Recommendations
If both train/test set are accurate as desired and the respective forecasting model repeatedly predicts a demand below zero, we would not recommend the product. If the model consistently forecasts a demand that is continuous with past performance or increases with desirable train/test results, we recommend the product. Certain forecasts of products could be improved with LSTM model's adjustment, but it is important to have a single model that allows us compare performance to each new product across the board to simulate time sensitive supply chain logistics.

![image](https://user-images.githubusercontent.com/47016027/89136287-e5c79500-d500-11ea-80a4-2d529bff5d32.png)

*This algorithm could be fine-tuned and improved even further to include: (a) more layers, (b) more epochs, and (c) different optimization methods
