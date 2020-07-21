import os
import unittest
from settings import configuration as config
from preprocess.file import uncompress, fileslist
from preprocess.prepare_netcdf import start_serialization


class TestPreprocess(unittest.TestCase):

    # test_uncompress
    file_tar_gz = 'd03_2017070900.tar.gz'
    nc_file = file = 'd03_2017070900/wrfout_d03_2017-07-09_04:00:00'

    def setUp(self):
        uncompress(os.path.join(config['TEST']['PREPROCESS'], self.file_tar_gz))

    def test_uncompress(self):
        self.assertTrue(os.path.exists(self.nc_file), 'El fichero {} se descomprimió.'.format(self.file_tar_gz))

    @unittest.skip('Not finished')
    def test_fileslist(self):
        self.assertIn(os.path.join(config['TEST']['PREPROCESS'], self.nc_file), fileslist())

    @unittest.skip('Not implemented yet')
    def test_uncompress_members_to_extract(self):
        pass

    @unittest.skip('Not implemented yet')
    # Test prepare NetCDF
    def test_start_serialization(self):
        """
        At least one .dat file should be in the uncompress directory path.
        """




if __name__ == '__main__':
    unittest.main()

