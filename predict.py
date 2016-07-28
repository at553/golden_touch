import pandas, numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler


class Predict:
    data = None

    def __init__(self, csvfile='data.csv'):
        dataframe = pandas.read_csv(csvfile, usecols=[1], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        self.data = dataset


    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)



    def train_model(self):
        # scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(self.data)

        # split into train and test sets
        train_size = int(len(dataset) * 0.95)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        look_back = 5
        trainX, trainY = self.create_dataset(train, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(6, input_dim=look_back))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
        return model


    def predict_new(self, input):
        model = self.train_model()
        assert len(input) == 5 and type(input) == list
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.data)
        inp = scaler.transform([input])
        print(scaler.inverse_transform(model.predict(numpy.array(inp).reshape(1, 1, 5))))


x = Predict()
x.predict_new([1243.068, 1298.713, 1336.560, 1299.175, 1288.913])
