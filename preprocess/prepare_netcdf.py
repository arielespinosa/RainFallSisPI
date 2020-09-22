import os
from datetime import datetime, timedelta
from multiprocessing import Process
import numpy as np

from settings import configuration as config
from files.netcdf import NetCDF
from preprocess.file import read_serialize_file, write_serialize_file
from .file import fileslist, move_files, uncompress

import threading
from shutil import rmtree


class SerializeFiles(threading.Thread):
    """
    Serialize all hourly nc file geted from SisPI dayly output uncompressed file
    """
    def __init__(self, file, path):
        threading.Thread.__init__(self)
        self.file = file
        self.path = path
        name      = self.file.split("/")[-1].replace("-", "_").split("_")
        self.name = "d_" + name[2] + name[3] + name[4] + name[5][:2] + ".dat"
        self.name = os.path.join(self.path, self.name)
       
    def run(self):
        sispi = NetCDF(self.file)
        sispi.vars(["Q2", "T2", "RAINC", "RAINNC"])
        sispi.save(self.name)


class UncompressFiles(threading.Thread):
    """
       Uncompress all SisPI tar.gz compressed file
    """
    def __init__(self, file=None, path=None, delete=True):
        threading.Thread.__init__(self)
        self.file = file
        self.path = path
        self.delete = delete
        self.threads = []

    def set_threads_ready(self):
        sispi_files = fileslist(self.path)
        # Prepare list of thread
        for file in sispi_files:
            path = file.split("/")[-2]
            path = os.path.join(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], path)
            self.threads.append(SerializeFiles(file, path))

    def remove_next_day_files(self):
        pass

    def run(self):
        i, c = 0, 0

        # Uncompress SisPI tar.gz file
        uncompress(self.file, self.path)

        # Prepare threads for read each nc file. One thread for file.
        self.set_threads_ready()

        for thread in self.threads:
            """
            Launch threads. 
            If threads in execution is more than 3 then wait for finshed at least 1
            """
            if c > 2:
                self.threads[i - 3].join()
                self.threads[i - 2].join()
                self.threads[i - 1].join()
                c = 0
            thread.start()

            c += 1
            i += 1

        # Remove nc uncompresed temp folder when serialization process end
        if self.delete:
            rmtree(self.path)


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
            for folder in os.listdir(config['DIRS']['SISPI_OUTPUT_DIR']):
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
    def export_sispi_lat_lon(cls, file):
        # 4th thing to do
        sispi = NetCDF(file)
        np.savetxt(config['SISPI_LAT'], sispi.xlat, delimiter=",", fmt="%7.2f")
        np.savetxt(config['SISPI_LON'], sispi.xlon, delimiter=",", fmt="%7.2f")

    @classmethod
    def sispi_rain_to_hour(cls):        
        files = fileslist(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'])    
        t = datetime(2017, 1, 1)

        while t.day is 1:
            sispi_t = read_serialize_file(
                os.path.join(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], "d_{}.dat".format(t.strftime('%Y%m%d%H')))
            )

            if t.hour is not 0:
                t_aux = t-timedelta(hours=1)

                sispi_t1 = read_serialize_file(
                    os.path.join(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], "d_{}.dat".format(t_aux.strftime('%Y%m%d%H')))
                )

                rain_sispi = np.asmatrix(sispi_t['RAINNC']-sispi_t1['RAINNC'])
                data = {
                    'Q2':sispi_t['Q2'],
                    'T2':sispi_t['T2'],
                    'RAIN_SISPI': rain_sispi,
                }         
                
            else:
                data = {
                    'Q2':sispi_t['Q2'],
                    'T2':sispi_t['T2'],
                    'RAIN_SISPI': sispi_t['RAINNC'],
                }

            write_serialize_file(data, os.path.join(config['DIRS']['SISPI_HOURLY_OUTPUT_DIR'], "d_{}.dat".format(t.strftime('%Y%m%d%H'))))

            t = t+timedelta(hours=1)

            
