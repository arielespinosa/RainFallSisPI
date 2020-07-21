from prepare_data import PrepareData as pdata
from preprocess.data_exploration import ExploratoryAnalisys as ea
from rna.train_models import TrainModelsExperiment as tm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main_interface():
    task = None

    while task is not 9:
        print("Seleccione una tarea a realizar:")
        print("1 - Serializar los datos de los ficheros NetCDF.")
        print("2 - Extraer los datos de GPM en formato horario.")
        print("3 - Interpolar SisPI y GPM e unirlos en un nuevo dataset.")
        print("4 - Seleccionar el área de La Habana.")
        print("5 - Realizar un análisis exploratorio de los datos sobre el área de La Habana.")
        print("6 - Preparar la serie de tiempo para el punto Norte")
        print("7 - Entrenar modelo tipo MLP.")

        print("9 - Salir del programa.")

        task = input()

        if int(task) is 1:
            pdata.prepare_sispi()
        elif int(task) is 2:
            pdata.prepare_gpm()
        elif int(task) is 3:
            pdata.combine_sispi_and_gpm()
        elif int(task) is 4:
            pdata.set_dataset_habana()
        elif int(task) is 5:
            files_with_nan = ea.files_with_nan_values_in_dataset()
            files_with_neg = ea.files_with_negative_values()
            min_max_val = ea.find_min_max_value()

            if len(files_with_nan) > 0:
                print("Ficheros con valores nulos:\n")
                print(files_with_nan)
            else:
                print("No hay valores nulos en la base de datos")

            if len(files_with_neg) > 0:
                print("Ficheros con valores negativos:\n")
                print(files_with_neg)
            else:
                print("No hay valores negativos en la base de datos")

            print("Valores extremos:\n")
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

        elif int(task) is 6:
            pdata.prepare_train_dataset("Norte.txt", point=(8, 8))

        elif int(task) is 7:
            tm.run()
            pass


        elif int(task) is 9:
            break
        else:
            continue

        print("Presione cualquier tecla para continuar...")
        input()
        
        

        # Clean prompt


if __name__ == '__main__':
    main_interface()



