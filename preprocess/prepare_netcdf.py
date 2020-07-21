import os
from multiprocessing import Process
import numpy as np
from files.netcdf import NetCDF
from .file import fileslist, move_files
from settings import configuration as config
from .extract_netcdf import UncompressFiles


class PrepareNetCDF:

    @classmethod
    def start_serialization(cls, _continue=False):
        """
        1st thing to do:

        Uncompress all tar.gz files from SisPI dir. This process run on threads for make fastest execution time.
        After extraction process, all extracted nc files are readed and serialized in a python dictionary structure
        where each key represent the var extracted from nc file and its value the 2D numpy array assigned.
        """

        i, c = 0, 0
        wrf_threads = []
        sispi_files = fileslist(config['DIRS']['SISPI_DIR'], searchtopdown=True, name_condition="d03")

        # Compare serialized and unserialize directories. Remove serialized files from files list to serialize
        # If is the firs time, its not necessary.
        if _continue:
            for folder in os.listdir(config['DIRS']['PREDICT_DATASET']):
                file = config['DIRS']['SISPI_DIR'] + "/" + folder + "/d03/d03.tar.gz"
                file = os.path.abspath(file)

                if file in sispi_files:
                    sispi_files.remove(file)

        for file in sispi_files:
            path = file.split('/')[-1].split('.')[0]
            path = os.path.join(config['DIRS']['SISPI_OUTPUT_DIR'], path)
            wrf_threads.append(UncompressFiles(file, path))

        for thread in wrf_threads:
            if c > 4:
                msg = "Progreso----------------------------------" + str(float(i / len(wrf_threads)) * 100) + " %"
                print(msg)
                wrf_threads[i - 5].join()
                wrf_threads[i - 4].join()
                wrf_threads[i - 3].join()
                wrf_threads[i - 2].join()
                wrf_threads[i - 1].join()
                c = 0

            thread.start()
            c += 1
            i += 1

    @classmethod
    def remove_not_required_files(cls):
        # 2nd thing to do

        files = fileslist(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], searchtopdown=True)

        for file in files:
            f = file.split("/")
            folder = f[-2]
            filename = f[-1].split("_")[-1].split(".")[0]

            if folder[-4:-2] != filename[-4:-2]:
                os.remove(file)

    @classmethod
    def put_sispi_files_on_right_path(cls):
        # 3rd thing to do
        files = fileslist(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], searchtopdown=True)
        i = int(len(files)/6)

        l, c = 0, 0
        process = [Process(target=move_files, args=(files[i*j: i*(j+1)], config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR']))
                   for j in range(6)]

        for p in process:
            if c > 2:
                process[l - 3].join()
                process[l - 2].join()
                process[l - 1].join()
                c = 0
            p.start()

            c += 1
            l += 1

        #os.removedirs(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'])
        #shutil.rmtree("/home/maibyssl/Ariel/rain/SisPIRainfall/data/untitled folder")

    @classmethod
    def export_sispi_lat_lon(cls, file="test/wrf.nc"):
        # 4th thing to do
        sispi = NetCDF(file)
        np.savetxt(config['SISPI_LAT'], sispi.xlat, delimiter=",", fmt="%7.2f")
        np.savetxt(config['SISPI_LON'], sispi.xlon, delimiter=",", fmt="%7.2f")
