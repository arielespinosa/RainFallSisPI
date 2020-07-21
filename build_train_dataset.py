import numpy as np
from preprocess.file import read_serialize_file, fileslist
from settings import configuration as config

"""
time_series("data/data_base/ts_5_west_point.txt", 5, (138, 106))
time_series("data/data_base/ts_5_east_point.txt", 5, (138, 130))
time_series("data/data_base/ts_5_north_point.txt", 5, (145, 118))
time_series("data/data_base/ts_5_center_point.txt", 5, (138, 118))

Por descubrir pero de no hacerlo poner y entrenar con:
time_series("data/data_base/ts_5_center_point.txt", 5, (138, 118))
"""

def time_series(fname, series_len, point=(0, 0), dataset_is_complete=True):
    dataset, series = [], []
    first_line = True
    count = 0

    dataset_habana = fileslist("/home/maibyssl/Ariel/rain/ariel/data/full_dataset")
    #dataset_habana = fileslist(config['DIRS']['DATASET_HABANA'])
    dataset_habana.sort()

    for file in dataset_habana:
        data = read_serialize_file(file)

        if dataset_is_complete:
            # If dataset have all days-hours its not necesary read each file
            if first_line:
                for i in range(series_len):
                    aux_data = read_serialize_file(dataset_habana[count-i])
                    value = aux_data['RAIN_SISPI'][point[0], point[1]]
                    series.insert(0, value)        
                first_line = False    
            else:
                series = dataset[count-1][1:series_len]
                series.append(data['RAIN_SISPI'][point[0], point[1]])
            
            series.append(data['RAIN_GPM'][point[0], point[1]])
            dataset.append(series)
            count += 1

        else:            
            # Preguntar a Maybis por la implementacion que hice si la incluyo o no
            pass

    if fname:
        np.savetxt(fname, dataset, delimiter=',', fmt='%7.2f')
    
    return dataset

if __name__ == "__main__":
    time_series("data/data_base/ts_5_center_point.txt", 5, (138, 118))
    pass