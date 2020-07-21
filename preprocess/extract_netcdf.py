import threading
from shutil import rmtree
from files.netcdf import *
from preprocess.file import fileslist, uncompress
from settings import configuration as config

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


# Uncompress tar.bz SisPI dayly output
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

