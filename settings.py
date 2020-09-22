import os
import pytz

# Main dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR_PATH = os.path.join(BASE_DIR, "rna/logs/")

configuration = {
    'DIRS': {
        # Where compresed SisPI files are stored
        'SISPI_DIR': "/home/maibyssl/Ariel/Data/SisPI", 

        # Where uncompress SisPI tar.gz files before serialization.
        'SISPI_OUTPUT_DIR': os.path.join(BASE_DIR, "data/sispi_output"),

        # Where store serialized SisPI nc files
        'SISPI_SERIALIZED_OUTPUT_DIR': os.path.join(BASE_DIR, "data/sispi_serialized_output"),

        # Where store serialized SisPI hourly rain data. RAINNC changed to RAIN_SISPI.
        'SISPI_HOURLY_OUTPUT_DIR': os.path.join(BASE_DIR, "data/sispi_hourly"),

        # Where GPM files are stored
        'GPM_DIR': "/home/maibyssl/Ariel/Data/GPM_2017",

        # Where store serialized GPM hdf5 files
        'GPM_OUTPUT_DIR': os.path.join(BASE_DIR, "data/gpm"),

        # Where store interpolated GPM data to SisPI (all grid)
        'GPM_INTERPOLATED': os.path.join(BASE_DIR, "data/gpm_interpolated"),

        # Where store SisPI and GPM interpolated data
        'DATASET': os.path.join(BASE_DIR, "data/dataset"),

        # Where store habana section from 'DATASET'.
        'DATASET_HABANA': os.path.join(BASE_DIR, "data/dataset_habana"),
        'DATASET_HABANA_TXT': os.path.join(BASE_DIR, "data/dataset_habana_txt"),

        # Where store point choices for train models
        'TRAIN_DATASET': os.path.join(BASE_DIR, "data/train_dataset"),

        # Where store trained models
        'MODELS_OUTPUT': os.path.join(BASE_DIR, "rna/outputs"),
    },
    
    # SisPI & GPM lat-long data
    'SISPI_LAT': os.path.join(BASE_DIR, "data/sispi_lat.txt"),
    'SISPI_LON': os.path.join(BASE_DIR, "data/sispi_lon.txt"),
    'GPM_LAT': os.path.join(BASE_DIR, "data/gpm_lat.txt"),
    'GPM_LON': os.path.join(BASE_DIR, "data/gpm_lon.txt"),
    
    # GPM example file for get metadata
    'GPM_FILE':'3B-HHR.MS.MRG.3IMERG.20170101-S000000-E002959.0000.V06B.HDF5',

    # SisPI example file for get metadata
    'SISPI_FILE':'wrfout_d03_2017-01-01_00:00:00',
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





