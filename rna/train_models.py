import os
import json
import numpy as np

# import Sklearn library
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# import Keras library
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam, Adagrad

# own modules
import settings as config
from preprocess.file import write_serialize_file
from .keras_models import MLP, LSTMNN, NSVM, ELM, CONV
from .rna_models_plots import plot_history, plot_scatter_results, plot_histogram_results
from .grid_predictions import predict_month, predict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

#datasets = ["data/data_base/ts_west_point.txt", "data/data_base/ts_north_point.txt", "data/data_base/ts_south_point.txt", "data/data_base/ts_east_point.txt", "data/data_base/ts_center_point.txt"]
datasets = ["data/data_base/ts_5_west_point.txt", "data/data_base/ts_5_north_point.txt", "data/data_base/ts_5_south_point.txt", "data/data_base/ts_5_east_point.txt", "data/data_base/ts_5_center_point.txt"]

# Neuronal Networks Models ------------------------------------------------------------------------
dense_units = [[10], [15], [20], [5, 5], [5, 10], [10, 10], [10, 15], [10, 20], [15, 20], [20, 20], [160, 80]]
#elm_dense_units = [300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
#nsvm_dense_units = [(5, [5]), (5, [10]), (5, [15]), (5, [5, 5]), (5, [5, 10]), (5, [10, 10]), (5, [10, 20]), (5, [20, 20]), (5, [160, 80])]

h_activation = ["sigmoid", "relu", "softsign"]

parameters = { 
        "dense_units"   : [600, 800],
        "h_activation"  : "sigmoid",
        "o_activation"  : "sigmoid",
        "batch_norm"    : [(0.99, 0.001), (0.99, 0.001), (0.99, 0.001)],
        "dropout"       : "d",
        "dropout_rate"  : [0.2, 0.1, 0.2],
        "optimizer"     : SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), 
        "loss"          : "mse",
        "metrics"       : ["mse", "mae"],
        "shape"         : (5, ),
        "name"          : "",
        "callbacks": [EarlyStopping(monitor="mse", patience=3),
                                ReduceLROnPlateau(monitor="mse", factor=0.2, patience=2, min_lr=0.001),                       
                                TerminateOnNaN()],
        "kernel_initializer": "",
        "shuffle": False,
        "epochs": 3,
        "batch_size": 500,
        "validation_split": 0.1,
        "dataset": None,
        }

class TrainModelsExperiment:

        @classmethod
        def train_mlp(cls, parameters, series_len, test_size=0.1):
                # Creating folder
                folder = os.path.join(config.BASE_DIR, "rna/outputs/{}".format(parameters["name"]))

                try:
                        os.mkdir(folder)
                except FileExistsError:
                        pass

                # Loading dataset
                dataset = np.loadtxt(parameters["dataset"], delimiter=',')

                # Preparing scalers for normalization
                x_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))

                # Normalizing dataset
                X = x_scaler.fit_transform(dataset[:, :series_len])
                Y = y_scaler.fit_transform(dataset[:, -1].reshape(-1, 1))

                # Spliting data for training and validation
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

                # Creating ANN model
                model = MLP(parameters)
                
                # Training & evaluating model
                model.train(x_train, y_train, validation_data=(x_test, y_test),
                        validation_split=parameters["validation_split"], 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        shuffle=parameters["shuffle"])

                # Making some predictions
                prediction = model.predict(x_test)
                predictions = y_scaler.inverse_transform(prediction)

                # Saving model and history
                model.save()
                model.save_history()

                # Saving parameters. Callbacks can't be serialized also SGD optimizer. Any way, always we use same config for it.
                with open(os.path.join(folder, 'parameters.json'), 'w') as json_file:
                        parameters["callbacks"] = ""
                        if not isinstance(parameters["optimizer"], str):
                                parameters["optimizer"] = "SGD"

                        json.dump(parameters, json_file)

                # Saving scalers
                write_serialize_file(x_scaler, "rna/outputs/{}/x_scaler.dat".format(parameters["name"]))
                write_serialize_file(y_scaler, "rna/outputs/{}/y_scaler.dat".format(parameters["name"]))
                
                # Inversing x_test & y_test for plot
                x_test = x_scaler.inverse_transform(x_test)
                y_test = y_scaler.inverse_transform(y_test)

                # Saving SisPi, GPM and ANN predictions inversed
                np.savetxt("rna/outputs/{}/sispi.txt".format(parameters["name"]),x_test[:, -1], delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/gpm.txt".format(parameters["name"]),y_test, delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/rna.txt".format(parameters["name"]), predictions, delimiter=',', fmt="%7.3f")

                # Making some plots
                plot_histogram_results(parameters["name"])
                plot_histogram_results(parameters["name"], False)
                plot_scatter_results(parameters["name"])
                plot_history(parameters["name"])

        @classmethod
        def train_lstm(cls, parameters, series_len, test_size=0.05):
                # Creating folder
                folder = os.path.join(config.BASE_DIR, "rna/outputs/{}".format(parameters["name"]))
                try:
                        os.mkdir(folder)
                except FileExistsError:
                        pass

                # Loading dataset
                dataset = np.loadtxt(parameters["dataset"], delimiter=',')

                # Preparing scalers for normalization
                x_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))

                # Normalizing dataset
                X = x_scaler.fit_transform(dataset[:, :series_len])
                Y = y_scaler.fit_transform(dataset[:, -1].reshape(-1, 1))

                # Spliting data for training and validation
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)

                # Reshaping x train for lstm (examples, ts_len, features)
                shape   = x_train.shape
                x_train = x_train.reshape(shape[0], shape[1], 1)

                shape  = x_test.shape
                x_test = x_test.reshape(shape[0], shape[1], 1)

                # Creating ANN model
                model = LSTMNN(parameters)
                
                # Training & evaluating model
                model.train(x_train, y_train, validation_data=(x_test, y_test), 
                        validation_split=parameters["validation_split"], 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        shuffle=parameters["shuffle"])
                
                # Making some predictions
                prediction = model.predict(x_test)
                predictions = y_scaler.inverse_transform(prediction)

                # Saving model and history
                model.save()
                model.save_history()

                # Saving parameters. Callbacks can't be serialized also SGD optimizer. Any way, always we use same config for it.
                with open(os.path.join(folder, 'parameters.json'), 'w') as json_file:
                        parameters["callbacks"] = ""
                        if not isinstance(parameters["optimizer"], str):
                                parameters["optimizer"] = "SGD"
                        json.dump(parameters, json_file)

                # Saving scalers
                write_serialize_file(x_scaler, "rna/outputs/{}/x_scaler.dat".format(parameters["name"]))
                write_serialize_file(y_scaler, "rna/outputs/{}/y_scaler.dat".format(parameters["name"]))
                
                # Inversing x_test & y_test for plot
                x_test = x_test.reshape(shape[0], series_len)
                x_test = x_scaler.inverse_transform(x_test)
                y_test = y_scaler.inverse_transform(y_test)

                # Saving SisPi, GPM and ANN predictions inversed
                np.savetxt("rna/outputs/{}/sispi.txt".format(parameters["name"]),x_test[:, -1], delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/gpm.txt".format(parameters["name"]),y_test, delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/rna.txt".format(parameters["name"]), predictions, delimiter=',', fmt="%7.3f")

                # Making some plots
                plot_histogram_results(parameters["name"])
                plot_histogram_results(parameters["name"], False)
                plot_scatter_results(parameters["name"])
                plot_history(parameters["name"])

        @classmethod
        def train_nsvm(cls, parameters, series_len, test_size=0.1):
                
                # Creating folder
                folder = os.path.join(config.BASE_DIR, "rna/outputs/{}".format(parameters["name"]))
                try:
                        os.mkdir(folder)
                except FileExistsError:
                        pass

                # Loading dataset
                dataset = np.loadtxt(parameters["dataset"], delimiter=',')

                # Preparing scalers for normalization
                x_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))

                # Normalizing dataset
                X = x_scaler.fit_transform(dataset[:, :series_len])
                Y = y_scaler.fit_transform(dataset[:, -1].reshape(-1, 1))

                # Spliting data for training and validation
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=parameters["shuffle"])

                # Three inputs
                _x_train = {
                "input_1":x_train[:, 0],
                "input_2":x_train[:, 1], 
                "input_3":x_train[:, 2],
                }

                _x_test = {
                "input_1":x_test[:, 0],
                "input_2":x_test[:, 1], 
                "input_3":x_test[:, 2],
                }

                # Creating ANN model
                model = NSVM(parameters)
        
                # Training & evaluating model
                model.train(_x_train, y_train, validation_data=(_x_test, y_test),
                        validation_split=parameters["validation_split"], 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        shuffle=parameters["shuffle"])
                
                # Making some predictions
                prediction = model.predict(_x_test)
                predictions = y_scaler.inverse_transform(prediction)

                # Saving model and history
                model.save()
                model.save_history()

                # Saving parameters. Callbacks can't be serialized also SGD optimizer. Any way, always we use same config for it.
                with open(os.path.join(folder, 'parameters.json'), 'w') as json_file:
                        parameters["callbacks"] = ""
                        if not isinstance(parameters["optimizer"], str):
                                parameters["optimizer"] = "SGD"
                        json.dump(parameters, json_file)

                # Saving scalers
                write_serialize_file(x_scaler, "rna/outputs/{}/x_scaler.dat".format(parameters["name"]))
                write_serialize_file(y_scaler, "rna/outputs/{}/y_scaler.dat".format(parameters["name"]))
                
                # Inversing x_test & y_test for plot
                x_test = x_scaler.inverse_transform(x_test)
                y_test = y_scaler.inverse_transform(y_test)

                # Saving SisPi, GPM and ANN predictions inversed
                np.savetxt("rna/outputs/{}/sispi.txt".format(parameters["name"]),x_test[:, -1], delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/gpm.txt".format(parameters["name"]),y_test, delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/rna.txt".format(parameters["name"]), predictions, delimiter=',', fmt="%7.3f")

                # Making some plots
                plot_histogram_results(parameters["name"])
                plot_histogram_results(parameters["name"], False)
                plot_scatter_results(parameters["name"])
                plot_history(parameters["name"])

        @classmethod
        def train_elm(cls, parameters, series_len, test_size=0.1):
                # Creating folder
                folder = os.path.join(config.BASE_DIR, "rna/outputs/{}".format(parameters["name"]))
                try:
                        os.mkdir(folder)
                except FileExistsError:
                        pass

                # Loading dataset
                dataset = np.loadtxt(parameters["dataset"], delimiter=',')

                # Preparing scalers for normalization
                x_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))

                # Normalizing dataset
                X = x_scaler.fit_transform(dataset[:, :series_len])
                Y = y_scaler.fit_transform(dataset[:, -1].reshape(-1, 1))

                # Spliting data for training and validation
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

                model = ELM(parameters)
        
                # Training & evaluating model
                model.train(x_train, y_train, validation_data=(x_test, y_test),
                        validation_split=parameters["validation_split"], 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        shuffle=parameters["shuffle"])

                # Making some predictions
                prediction = model.predict(x_test)
                predictions = y_scaler.inverse_transform(prediction)

                # Saving model and history
                model.save()
                model.save_history()

                # Saving parameters. Callbacks can't be serialized also SGD optimizer. Any way, always we use same config for it.
                with open(os.path.join(folder, 'parameters.json'), 'w') as json_file:
                        parameters["callbacks"] = ""
                        if not isinstance(parameters["optimizer"], str):
                                parameters["optimizer"] = "SGD"
                        json.dump(parameters, json_file)

                # Saving scalers
                write_serialize_file(x_scaler, "rna/outputs/{}/x_scaler.dat".format(parameters["name"]))
                write_serialize_file(y_scaler, "rna/outputs/{}/y_scaler.dat".format(parameters["name"]))
                
                # Inversing x_test & y_test for plot
                x_test = x_scaler.inverse_transform(x_test)
                y_test = y_scaler.inverse_transform(y_test)

                # Saving SisPi, GPM and ANN predictions inversed
                np.savetxt("rna/outputs/{}/sispi.txt".format(parameters["name"]),x_test[:, -1], delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/gpm.txt".format(parameters["name"]),y_test, delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/rna.txt".format(parameters["name"]), predictions, delimiter=',', fmt="%7.3f")

                # Making some plots
                plot_histogram_results(parameters["name"])
                plot_histogram_results(parameters["name"], False)
                plot_scatter_results(parameters["name"])
                plot_history(parameters["name"])

        @classmethod
        def train_conv(cls, parameters, series_len, test_size=0.1):

                # Creating folder
                folder = os.path.join(config.BASE_DIR, "rna/outputs/{}".format(parameters["name"]))
                try:
                        os.mkdir(folder)
                except FileExistsError:
                        pass

                # Loading dataset
                dataset = np.loadtxt(parameters["dataset"], delimiter=',')

                # Preparing scalers for normalization
                x_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))

                # Normalizing dataset
                X = x_scaler.fit_transform(dataset[:, :series_len])
                Y = y_scaler.fit_transform(dataset[:, -1].reshape(-1, 1))

                # Spliting data for training and validation
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                model = CONV(parameters)

                # Training & evaluating model
                model.train(x_train, y_train, validation_data=(x_test, y_test),
                        validation_split=parameters["validation_split"], 
                        batch_size=parameters["batch_size"], 
                        epochs=parameters["epochs"], 
                        shuffle=parameters["shuffle"])

                # Making some predictions
                prediction = model.predict(x_test)
                predictions = y_scaler.inverse_transform(prediction)

                # Saving model and history
                model.save()
                model.save_history()

                # Saving parameters. Callbacks can't be serialized also SGD optimizer. Any way, always we use same config for it.
                with open(os.path.join(folder, 'parameters.json'), 'w') as json_file:
                        parameters["callbacks"] = ""
                        if not isinstance(parameters["optimizer"], str):
                                parameters["optimizer"] = "SGD"
                        json.dump(parameters, json_file)

                # Saving scalers
                write_serialize_file(x_scaler, "rna/outputs/{}/x_scaler.dat".format(parameters["name"]))
                write_serialize_file(y_scaler, "rna/outputs/{}/y_scaler.dat".format(parameters["name"]))
                
                # Inversing x_test & y_test for plot
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
                x_test = x_scaler.inverse_transform(x_test)
                y_test = y_scaler.inverse_transform(y_test)

                # Saving SisPi, GPM and ANN predictions inversed
                np.savetxt("rna/outputs/{}/sispi.txt".format(parameters["name"]),x_test[:, -1], delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/gpm.txt".format(parameters["name"]),y_test, delimiter=',', fmt="%7.3f")
                np.savetxt("rna/outputs/{}/rna.txt".format(parameters["name"]), predictions, delimiter=',', fmt="%7.3f")

                # Making some plots
                plot_histogram_results(parameters["name"])
                plot_histogram_results(parameters["name"], False)
                plot_scatter_results(parameters["name"])
                plot_history(parameters["name"])

        @classmethod
        def run(cls):

                #ts_len = int(sys.argv[-1])

                ts_len = 5
                
                dataset = datasets[-1]
                number = 9999

                parameters["h_activation"] = h_activation[2]
                parameters["dense_units"] = dense_units[0]
                parameters["dataset"] = dataset

                if ts_len is 3:
                        name = dataset.split("_")[2] 
                        parameters["name"] = "tts_{0}_conv_model_{1}".format(name, number)
                elif ts_len is 5:
                        name = dataset.split("_")[2] 
                        name2 = dataset.split("_")[3]
                        parameters["name"] = "tts_{0}_{1}_conv_model_{2}".format(name, name2, number)

                cls.train_mlp(parameters, ts_len)
                #train_nsvm(parameters, ts_len)
                #train_lstm(parameters, ts_len)
                #train_conv(parameters, ts_len)
                print("Finish!!!")


class MakeGridPredictions:

        # Prepare data (time steps = 5) for make predictions on the Habana space grid
        @classmethod
        def __prepare_data(cls, year, month, day, hour):
                dataset = []

                for i in range(5):
                        file = "data/full_dataset_txt/d_%04d%02d%02d%02d.txt" % (year, month, day, hour - i)
                        data = np.loadtxt(file, delimiter=",")
                        dataset.insert(0, data[:, 2])

                data = np.stack(np.array(dataset), axis=1)
                np.savetxt("data/predict_grid/d_%04d%02d%02d%02d.txt" % (year, month, day, hour), data, fmt="%7.3f", delimiter=',')

        @classmethod
        def __predict_area(cls, data, area):
                predictions = []

                if area is "north":
                        x_scaler = read_serialize_file("rna/outputs/tts_5_north_mlp_model_7/x_scaler.dat")
                        y_scaler = read_serialize_file("rna/outputs/tts_5_north_mlp_model_7/y_scaler.dat")
                        points = x_scaler.transform(data[:125])

                        model = MLP("rna/outputs/tts_5_north_mlp_model_7/tts_5_north_mlp_model_7.h5")

                elif area is "east":
                        x_scaler = read_serialize_file("rna/outputs/tts_5_east_mlp_model_9/x_scaler.dat")
                        y_scaler = read_serialize_file("rna/outputs/tts_5_east_mlp_model_9/y_scaler.dat")

                        points = [data[i:i + 8] for i in range(125, 250, 25)]
                        points = np.array(points)
                        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
                        points = x_scaler.transform(points)

                        model = MLP("rna/outputs/tts_5_east_mlp_model_9/tts_5_east_mlp_model_9.h5")

                elif area is "center":
                        x_scaler = read_serialize_file("rna/outputs/tts_5_center_mlp_model_25/x_scaler.dat")
                        y_scaler = read_serialize_file("rna/outputs/tts_5_center_mlp_model_25/y_scaler.dat")

                        points = [data[i:i + 9] for i in range(132, 257, 25)]
                        points = np.array(points)
                        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
                        points = x_scaler.transform(points)

                        model = MLP("rna/outputs/tts_5_center_mlp_model_25/tts_5_center_mlp_model_25.h5")

                elif area is "west":
                        x_scaler = read_serialize_file("rna/outputs/tts_5_west_mlp_model_19/x_scaler.dat")
                        y_scaler = read_serialize_file("rna/outputs/tts_5_west_mlp_model_19/y_scaler.dat")

                        points = [data[i:i + 8] for i in range(142, 267, 25)]
                        points = np.array(points)
                        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
                        points = x_scaler.transform(points)

                        model = MLP("rna/outputs/tts_5_west_mlp_model_19/tts_5_west_mlp_model_19.h5")

                elif area is "south":
                        x_scaler = read_serialize_file("rna/outputs/tts_5_south_mlp_model_29/x_scaler.dat")
                        y_scaler = read_serialize_file("rna/outputs/tts_5_south_mlp_model_29/y_scaler.dat")
                        points = x_scaler.transform(data[250:])

                        model = MLP("rna/outputs/tts_5_south_mlp_model_29/tts_5_south_mlp_model_29.h5")

                results = model.predict(points)
                results = y_scaler.inverse_transform(results)

                del model
                return results

        @classmethod
        def __unify_predict_area(cls, north, east, center, west, south):
                n = north.reshape(5, 25)
                e = east.reshape(5, 8)
                c = center.reshape(5, 9)
                w = west.reshape(5, 8)
                s = south.reshape(5, 25)

                middle = np.hstack((e, c, w))
                return np.vstack((n, middle, s))

        @classmethod
        def __predict(cls, y, m, d, h):
                cls.__prepare_data(y, m, d, h)
                file = "d_%04d%02d%02d%02d.txt" % (y, m, d, h)

                try:
                        data = np.loadtxt("data/predict_grid/{}".format(file), delimiter=",")

                        n = cls.__predict_area(data, "north")
                        e = cls.__predict_area(data, "east")
                        c = cls.__predict_area(data, "center")
                        w = cls.__predict_area(data, "west")
                        s = cls.__predict_area(data, "south")
        
                        grid = cls.__unify_predict_area(n, e, c, w, s)

                        np.savetxt("rna/grid_predictions/{}".format(file), grid, fmt="%7.3f", delimiter=",")
                except:
                        pass

        @classmethod
        def predict_month(cls, y, m, d):
                t = datetime(y, m, d)

                while t.month == m:
                        p = Process(target=cls.__predict, args=(t.year, t.month, t.day, t.hour))
                        p.start()
                        p.join()

                        t += timedelta(hours=1)
