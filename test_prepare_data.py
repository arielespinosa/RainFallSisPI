import os
import unittest
from settings import configuration as config
from preprocess.prepare_netcdf import *
from preprocess.prepare_gpm import gpm_to_hourly, export_gpm_lat_lon
from preprocess.build_dataset import interpolate_gpm, join_sispi_and_gpm, select_habana_from_dataset

"""
How implement unit test for void methods?
"""

class TestPrepareData(unittest.TestCase):
    file_tar_gz = 'd03_2017070900.tar.gz'
    nc_file = file = 'd03_2017070900/wrfout_d03_2017-07-09_04:00:00'

    @unittest.skip('Check how wait for uncompress process stop before continue test')
    def test_start_serialization(self):
        dat_files_list = []
        dat_files = ["d_2017070900.dat", "d_2017070901.dat", "d_2017070903.dat",
                     "d_2017071010.dat", "d_2017071011.dat", "d_2017071012.dat"]

        for path, directories, files in  os.walk(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], topdown=True):
            dat_files_list.extend(files)

        self.assertEqual(len(dat_files_list), 0)
        start_serialization()

        for path, directories, files in os.walk(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'], topdown=True):
            dat_files_list.extend(files)

        for file in dat_files:
            self.assertIn(file, dat_files_list)


    @unittest.skip('Not implemented yet')
    def test_remove_not_required_files(self):
        pass


    @unittest.skip('Not implemented yet')
    def test_put_sispi_files_on_right_path(self):
        pass

    @unittest.skip('Not implemented yet')
    # Test prepare NetCDF
    def test_export_sispi_lat_lon(self):
        """
        At least one .dat file should be in the uncompress directory path.
        """


if __name__ == '__main__':
    unittest.main()

