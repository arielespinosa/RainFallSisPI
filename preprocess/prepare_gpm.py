import os
import numpy as np
from files.gpm import GPM
from preprocess.file import write_serialize_file
from settings import configuration as config

def gpm_filename(filename, as_dat = False):
    filename = filename.split("3IMERG.")[-1].split("-")
    if as_dat:
        filename = "d_" + filename[0] + filename[1][1:3] + ".dat"
    else:
        filename = "d_" + filename[0] + filename[1][1:3] + ".txt"
    return filename

def gpm_to_hourly(grid, gpm_files_list, save_as_dat_too = False):
    for i in range(1, len(gpm_files_list), 2):
        filename = os.path.join(config['DIRS']['GPM_OUTPUT_DIR'], gpm_filename(gpm_files_list[i]))

        file1 = GPM(gpm_files_list[i-1], grid)
        file2 = GPM(gpm_files_list[i], grid)
        hourly_rain = file1.rain() + file2.rain()

        np.savetxt(filename, hourly_rain, delimiter=",", fmt="%7.2f")
        
        if save_as_dat_too:
            write_serialize_file(hourly_rain, filename)

def export_gpm_lat_lon(file, grid):
    gpm = GPM(file, grid)
    gpm.save_lat_as_txt(config['GPM_LAT'])
    gpm.save_long_as_txt(config['GPM_LON'])
