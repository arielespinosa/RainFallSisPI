import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import SGD

from prepare_data import PrepareData as pdata
from preprocess.data_exploration import ExploratoryAnalisys as ea
from rna.train_models import (TrainModels as tm, ModelGridPrediction as mgp)
from rna.evaluate_models import EvaluateModels as em
from settings import configuration as config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ts_3_dataset = ["data/data_base/ts_west_point.txt", "data/data_base/ts_north_point.txt", "data/data_base/ts_south_point.txt", "data/data_base/ts_east_point.txt", "data/data_base/ts_center_point.txt"]
ts_5_dataset = ["data/data_base/ts_5_west_point.txt", "data/data_base/ts_5_north_point.txt", "data/data_base/ts_5_south_point.txt", "data/data_base/ts_5_east_point.txt", "data/data_base/ts_5_center_point.txt"]

#dense_units = [[10], [15], [20], [5, 5], [5, 10], [10, 10], [10, 15], [10, 20], [15, 20], [20, 20], [160, 80]]
#nsvm_dense_units = [(5, [5]), (5, [10]), (5, [15]), (5, [5, 5]), (5, [5, 10]), (5, [10, 10]), (5, [10, 20]), (5, [20, 20]), (5, [160, 80])]
#h_activation = ["sigmoid", "relu", "softsign"]

parameters = { 
    "dense_units"   : [10, 10],
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
    "epochs": 20,
    "batch_size": 500,
    "validation_split": 0.1,
    "dataset": ts_5_dataset[0],
}

menu = """
Seleccione una tarea a realizar:
     0 - Descomprimir los ficheros tar.gz de SisPI y serializar los NetCDF.
     1 - Convertir a horario los valores de lluvia en los datos de SisPI.
     2 - Extraer los datos de GPM en formato horario.
     3 - Interpolar SisPI y GPM e unirlos en un nuevo dataset.
     4 - Seleccionar el área de La Habana.
     5 - Realizar un análisis exploratorio de los datos sobre el área de La Habana.
     6 - Preparar la serie de tiempo para los distintos puntos.
     7 - Entrenar modelo modelo de red.
     8 - Calcular los valores estadísticos para cada modelo y llevarlos a CSV.
     9 - Hallar los mejores modelos por punto y por tipo.
    10 - Realizar predicciones para el área de estudio.
    11 - Llevar a csv las estadisticas de los modelos en la correccion para el area de estudio.
"""

def interface():
    task = int(input(menu))

    while task >= 0:      
        if task is 0:
            pdata.uncompress_sispi()
        elif task is 1:
            pdata.prepare_sispi()
        elif task is 2:
            pdata.prepare_gpm()
        elif task is 3:
            pdata.combine_sispi_and_gpm(interpolated=False)
        elif task is 4:
            pdata.set_dataset_habana()
        elif task is 5:
            files_with_nan = ea.files_with_nan_values_in_dataset()
            files_with_neg = ea.files_with_negative_values()
            min_max_val = ea.find_min_max_value()

            if len(files_with_nan) > 0:
                print("Ficheros con valores nulos:\n")
                print(files_with_nan)
            else:
                print("No hay valores nulos en la base de datos:\n")

            if len(files_with_neg) > 0:
                print("Ficheros con valores negativos:\n")
                print(files_with_neg)
            else:
                print("No hay valores negativos en la base de datos:\n")

            print("Valores extremos:")
            print("Variable: Q2 | valor minimo: %04f | valor máximo: %04f ." % (
                min_max_val["values"]["minQ2"], min_max_val["values"]["maxQ2"])
                )
            print("Variable: T2 | valor minimo: %04f | valor máximo: %04f ." % (
                min_max_val["values"]["minT2"], min_max_val["values"]["maxT2"])
                )
            print("Variable: RAIN_SISPI | valor máximo: %04f ." % (
                min_max_val["values"]["maxRAIN_SISPI"])
                )
            print("Variable: RAIN_GPM | valor máximo: %04f ." % (
                min_max_val["values"]["maxRAIN_GPM"])
                )
        elif task is 6:
            pdata.prepare_train_dataset("Norte.txt", series_len=5, point=(8, 8))
        elif task is 7:
            tm.run(model_type='mlp', ts_len=5, number=99999, parameters=parameters)
        elif task is 8:
            em.model_statistics(config['DIRS']['MODELS_OUTPUT'])
            em.statistics_to_csv(config['DIRS']['MODELS_OUTPUT'], "pruebammm.csv")
        elif task is 9:
            em.best_models_by_type_and_point(config['DIRS']['MODELS_OUTPUT'], "prueba.txt")
        elif task is 10:
            mgp.predict(2017, 6, 1, 22)
            em.eval_models_grid_predictions(2017, 6, 1, 22)
        elif task is 11:
            em.models_grid_statistics_to_csv("csvf.csv")
            pass
        elif task is 12:
            pass
        else:
            pass
        
        task = int(input(menu))

if __name__ == '__main__':
    interface()
  
