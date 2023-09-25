
import pandas as pd
import numpy as np
from numpy import asarray
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder#robust scaler
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from sklearn.metrics import mean_squared_error
import math

USA = pd.read_csv(r'C:\Users\chateaux.m\Documents\DESU_DS\Projet\Data\USA\USA_f.csv')
USA.drop(['Unnamed: 0'], axis = 1, inplace = True)

PRED = []
REAL = []
statei = 1
for state in USA['State'].unique():
# state = "FLORIDA"
	USA_florida = USA[USA['State']==state]
	USA_florida = USA_florida.drop(['State'],axis = 1)
	targetVar_idx = np.where(USA_florida.columns == 'Cases_per_month')[0][0]
	# USA_florida = USA_florida.drop(['State','Year','Month'],axis = 1)
	# USA_florida = USA_florida[USA_florida['sequential_months']<= 8*12]

	# # one hot encoder 
	# encoder = OneHotEncoder(handle_unknown='ignore')
	# #perform one-hot encoding on 'embarked' column
	# encoded = pd.DataFrame(encoder.fit_transform(USA_florida[['Year','Month','sequential_months']]).toarray(), columns = encoder.get_feature_names_out())
	# USA_florida = USA_florida.drop(['Year','Month','sequential_months'],axis = 1)
	# USA_florida = USA_florida.join(encoded)

	print(USA_florida.head())

	values = USA_florida.values
	# ensure all data is float
	values = values.astype('float32')

	# normalize features
	ValuesWO_target = np.delete(values, targetVar_idx, axis=1)
	scaler =  RobustScaler().fit(ValuesWO_target)
	values_scaled = scaler.transform(ValuesWO_target)
	values_scaled = np.column_stack((values_scaled,values[:,targetVar_idx]))


	# convert series to supervised learning
	def series_to_supervised(data, n_in = 1, n_out = 1, dropnan=False):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg


	# frame as supervised learning
	past_length = 1
	reframed = series_to_supervised(values_scaled, past_length, 1)
	print(reframed.shape)

	# cut data bw january 2014 and december 2021
	jan_2014_idx = np.where((USA_florida['Year']==2014) & (USA_florida['Month']==1))[0][0]
	dec_2021_idx = np.where((USA_florida['Year']==2021) & (USA_florida['Month']==12))[0][0]

	reframed = reframed.iloc[jan_2014_idx:dec_2021_idx+1,:]
	print(reframed.shape)
	reframed = reframed.fillna(0)#remplacer nan des cases per month par des 0
	print(sum(np.array(pd.isna(reframed)))) 

	# reframed = reframed.iloc[0:8*12,:]
	# reframed.dropna(inplace=True)
	# print(reframed.shape)


	# split into train and test sets
	values = reframed.values
	n_train_months = 12*7 
	train = values[:n_train_months, :]
	test = values[n_train_months:, :]
	# split into input and outputs
	# input_indices = np.array(list(range(0,13*past_length+9))+ list(range(13*past_length+10,values.shape[1])))
	input_indices = np.array(list(range(0,13*past_length*2-1)))
	# output_index = 13*past_length+targetVar_idx
	output_index = train.shape[1]-1
	train_X, train_y = train[:,input_indices], train[:, output_index]
	test_X, test_y = test[:,input_indices], test[:, output_index]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



	# design network
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss='mae', optimizer='adam',run_eagerly=True)
	# fit network
	history = model.fit(train_X, train_y, epochs=50, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	# plot history
	# plt.plot(history.history['loss'], label='train')
	# plt.plot(history.history['val_loss'], label='test') #https://stackoverflow.com/questions/55746382/why-val-loss-and-val-acc-are-not-displaying
	# plt.legend()
	# plt.show()


	# make a prediction
	yhat = model.predict(test_X)
	yhat = yhat[:,0]
	# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	# # invert scaling for forecast
	# inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
	# # inv_yhat = scaler.inverse_transform(inv_yhat)
	# # inv_yhat = inv_yhat[:,0]
	# # # invert scaling for actual
	# # test_y = test_y.reshape((len(test_y), 1))
	# inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
	# # inv_y = scaler.inverse_transform(inv_y)
	# # inv_y = inv_y[:,0]
	# # calculate RMSE
	# # rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
	# rmse = math.sqrt(mean_squared_error(test_y, yhat))
	# print('Test RMSE: %.3f' % rmse)

	if statei == 1:
		PRED = yhat
		REAL = test_y
	else:
		PRED = np.column_stack((PRED, yhat))
		REAL = np.column_stack((REAL, test_y))

	statei+=1
	print(statei)






# ref = REAL[0:12]
t = np.arange(1,13)
obs = np.sum(REAL,axis = 1)
pred = np.sum(PRED,axis = 1)

bar_width = 0.35
plt.bar(t-bar_width,obs, width = bar_width, label = 'Cas observés', color = 'green', alpha = 0.7)
plt.bar(t ,pred, width = bar_width, label = 'Cas prédits', color = 'orange', alpha = 0.7)
# plt.bar(t + bar_width,nb_zeros, width = bar_width, label = 'Cas prédits', color = 'orange', alpha = 0.7)
plt.xticks(range(1,13),['Janvier','Février','Mars','Avril','Mai','Juin','Juillet','Aout','Septembre','Octobre','Novembre','Décembre'],rotation = 45)
plt.ylabel("Nombre de cas total sur l'ensemble des Etats-Unis")
plt.legend()
plt.show()




# # Define sequence length (e.g., 3 months for prediction)
# sequence_length = 12*7

# # Select relevant columns
# data = USA[['Temp_min', 'Temp_mean', 'Temp_max', 'Precipitation', 'Wind_speed', 'Cloud_cover','Pop_density','Humidity','Cases_per_month','sequential_months']]
# data = data[data['sequential_months']<=12*8]

# # Normalize numerical features (optional but recommended)
# scaler = RobustScaler()
# data[['Temp_min', 'Temp_mean', 'Temp_max', 'Precipitation', 'Wind_speed', 'Cloud_cover','Pop_density','Humidity']] = scaler.fit_transform(data[['Temp_min', 'Temp_mean', 'Temp_max', 'Precipitation', 'Wind_speed', 'Cloud_cover','Pop_density','Humidity']])

# # Create sequences and corresponding target values
# X = []
# y = []

# #sequence matrix: (sequence_length*Nstates)*Nvariables

# for i in range(int(len(data)/len(USA['State'].unique()) - sequence_length)):

#     X.append(data.loc[(data['sequential_months']>=i+1) & (data['sequential_months']<=i+sequence_length), ['Temp_min', 'Temp_mean', 'Temp_max', 'Precipitation', 'Wind_speed', 'Cloud_cover','Pop_density','Humidity']].values)  # Features (Climate and Demographics)
#     y.append(data.loc[(data['sequential_months']>=i+1) & (data['sequential_months']<=i+sequence_length), 'Cases_per_month'])  # Target (Dengue Cases)
    

# X = np.array(X)
# num_features = X.shape[2]
# y = np.array(y)


# # Define the RNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length,len(USA['State'].unique()), num_features)),
#     tf.keras.layers.Dense(50, activation = 'softmax')  # Output layer for regression
#     # Dense(50, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Train-test split
# # X_train = X[0:9,:,:]
# # X_test = X[10:11,:,:]
# # y_train = y[0:9,:]
# # y_test = y[10:11,:] 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Train the model
# model.fit(X_train, y_train, epochs=5, batch_size=100)

# # Evaluate the model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = np.mean(np.abs(y_test - y_pred))

# # Make predictions for the year 2021
# # predictions= model.predict(X_test)