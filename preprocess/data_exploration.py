import numpy as np
from settings import configuration as config
from preprocess.file import fileslist, read_serialize_file, write_serialize_file

def find_min_max_values(dataset):
    data = dict()
    for key in dataset.keys():
        var_key = dict({ key:{"min": np.amin(dataset[key]), "max":np.amax(dataset[key])} })      
        data.update(var_key)                                                                                                           
        del var_key
    return data

class ExploratoryAnalisys:

    @classmethod
    def files_with_nan_values_in_dataset(cls):
        files_with_nan_values = []

        for file in fileslist(config['DIRS']['DATASET_HABANA']):
            data = read_serialize_file(file)
            if np.isnan(data['RAIN_GPM']).any() or np.isnan(data['RAIN_SISPI']).any():
                files_with_nan_values.append((file.split('/')[-1]))

        return files_with_nan_values

    @classmethod
    def files_with_negative_values(cls):
        files_with_negative_values = []

        for file in fileslist(config['DIRS']['DATASET_HABANA']):
            data = read_serialize_file(file)
            if np.isneginf(data['RAIN_GPM']).any() or np.isnan(data['RAIN_SISPI']).any():
                files_with_negative_values.append((file.split('/')[-1]))

        return files_with_negative_values

    @classmethod
    def find_min_max_value(cls):
        minQ2, maxQ2, minT2, maxT2, maxRAIN_SISPI, maxRAIN_GPM  = 10.0, 0.0, 500.0, 0.0, 0.0, 0.0
        day_maxRAIN_SISPI, day_maxRAIN_GPM = "", ""

        for file in fileslist(config['DIRS']['DATASET_HABANA']):

            data = find_min_max_values(read_serialize_file(file)) 

            if data["Q2"]["min"] < minQ2:        
                minQ2 = data["Q2"]["min"]
                day_minQ2 = file.split("_")[-1].split(".")[0] 

            if data["Q2"]["max"] > maxQ2:
                maxQ2 = data["Q2"]["max"]
                day_maxQ2 = file.split("_")[-1].split(".")[0]

            if data["T2"]["min"] < minT2:
                minT2 = data["T2"]["min"]
                day_minT2 = file.split("_")[-1].split(".")[0]

            if data["T2"]["max"] > maxT2:
                maxT2 = data["T2"]["max"]
                day_maxT2 = file.split("_")[-1].split(".")[0]

            if data["RAIN_SISPI"]["max"] > maxRAIN_SISPI:
                maxRAIN_SISPI = data["RAIN_SISPI"]["max"]
                day_maxRAIN_SISPI = file.split("_")[-1].split(".")[0]

            if data["RAIN_GPM"]["max"] > maxRAIN_GPM:
                maxRAIN_GPM = data["RAIN_GPM"]["max"]
                day_maxRAIN_GPM = file.split("_")[-1].split(".")[0]

        results = { 
            "values": {
                "minQ2": minQ2,
                "maxQ2": maxQ2,
                "minT2": minT2,
                "maxT2": maxT2,
                "minRAIN_SISPI": 0.0,
                "maxRAIN_SISPI": maxRAIN_SISPI,
                "minRAIN_GPM": 0.0,
                "maxRAIN_GPM": maxRAIN_GPM,
                },

            "days": {
                "day_minQ2": day_minQ2,
                "day_maxQ2": day_maxQ2,
                "day_minT2": day_minT2,
                "day_maxT2": day_maxT2,
                "day_maxRAIN_SISPI": day_maxRAIN_SISPI,
                "day_maxRAIN_GPM": day_maxRAIN_GPM,
                },
            }

        write_serialize_file(results, "outputs/min_max_values_in_dataset.dat")

        return results

