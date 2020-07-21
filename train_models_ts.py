import os
import sys
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam, Adagrad
from .train_models import TrainModels as tm


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
        "shape"         : (5, 1),
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
        
class TimeSeriesExperiment:

        @classmethod
        def run(cls)

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

                tm.train_mlp(parameters, ts_len)
                #train_nsvm(parameters, ts_len)
                #train_lstm(parameters, ts_len)
                #train_conv(parameters, ts_len)
                print("Finish!!!")

        


    

