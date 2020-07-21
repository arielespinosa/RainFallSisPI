import os
import pytz

# Main dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR_PATH = os.path.join(BASE_DIR, "rna/logs/")

configuration = {
    'DIRS': {
        'SISPI_DIR': os.path.join("test/preprocess"),
        'SISPI_OUTPUT_DIR': os.path.join(BASE_DIR, "data/temp_sispi_output"),
        'SISPI_SERIALIZED_OUTPUT_DIR': os.path.join(BASE_DIR, "data/sispi"),

        'DATASET': os.path.join(BASE_DIR, "data/dataset"),
        'DATASET_HABANA': os.path.join(BASE_DIR, "data/dataset_habana"),
        'TRAIN_DATASET': os.path.join(BASE_DIR, "data/train_dataset"),

        'GPM_DIR': "/home/maibyssl/Ariel/GPM_2017",
        'GPM_OUTPUT_DIR': os.path.join(BASE_DIR, "data/gpm"),
        'GPM_INTERPOLATED': os.path.join(BASE_DIR, "data/gpm_interpolated"),
    },
    'SISPI_LAT': os.path.join(BASE_DIR, "data/sispi_lat.txt"),
    'SISPI_LON': os.path.join(BASE_DIR, "data/sispi_lon.txt"),
    'GPM_LAT': os.path.join(BASE_DIR, "data/gpm_lat.txt"),
    'GPM_LON': os.path.join(BASE_DIR, "data/gpm_lon.txt"),

    'TEST': {
        'PREPROCESS': os.path.join(BASE_DIR, "test/preprocess"),
    },

}


# Datasets dirs
FULL_DATASET   = os.path.join(BASE_DIR, "data/full_dataset")
PREDICT_DATASET = os.path.join(BASE_DIR, "data/full_dataset")
HOURLY_TRAIN_DATASET   = os.path.join(BASE_DIR, "data/hourly_train_dataset")
HOURLY_PREDICT_DATASET = os.path.join(BASE_DIR, "data/hourly_predict_dataset")
TS_GPM_TRAIN_DATASET = os.path.join(BASE_DIR, "data/gpm_six_time_steps_dataset")
TS_GPM_PREDICT_DATASET = os.path.join(BASE_DIR, "data/gpm_six_time_predict_dataset")
TRAIN_DATASET_HABANA   = os.path.join(BASE_DIR, "data/dataset_habana")
PREDICT_DATASET_HABANA = os.path.join(BASE_DIR, "data/predict_dataset_habana")



RNA_RESULTS = os.path.join(BASE_DIR, "rna/results")


# Time zones
TZ_CUBA = pytz.timezone('America/Bogota')
TZ_GMT0 = pytz.timezone('Etc/GMT-0')

# Grid specifications
SISPI_GRID  = {"lat": 183, "long": 411}
CMORPH_GRID = {"lat": 183, "long": 411}


# RNA specifications





