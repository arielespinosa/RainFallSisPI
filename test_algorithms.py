import os
import unittest
from pyunitreport import HTMLTestRunner
from settings import configuration as config
from prepare_data import PrepareData
from preprocess.data_exploration import ExploratoryAnalisys as ea
"""
How implement unit test for void methods?
"""

class TestPrepareData(unittest.TestCase):
    file_tar_gz = 'd03_2017070900.tar.gz'
    nc_file = file = 'd03_2017070900/wrfout_d03_2017-07-09_04:00:00'

    @unittest.skip('Not required by the client')
    def test_prepare_sispi(self):
        self.assertEquals(2+4, 6)       

    @unittest.skip('Not required by the client')
    def test_prepare_gpm(self):
        pass

    @unittest.skip('Not implemented yet')
    def test_combine_sispi_and_gpm(self): 
        pass       

    @unittest.skip('Not implemented yet')
    def test_set_dataset_habana(self):
        pass
    
    @unittest.skip('Not implemented yet')
    def test_prepare_train_dataset(self):
        pass


class TestExploratoryAnalisys(unittest.TestCase):

    def test_files_with_nan_values_in_dataset(self):
        files = ea.files_with_nan_values_in_dataset('test/data/')
        self.assertEquals(files[0], 'd_2017070900.dat')       

    def test_files_with_negative_values(self):
        files = ea.files_with_negative_values('test/data/')
        self.assertEquals(files[0], 'd_2017070900.dat')      

    def test_find_min_max_value(self):
        v = {
            'maxQ2': 0.021, 
            'maxT2': 301.952, 
            'maxRAIN_GPM': 2.9, 
            'minQ2': 0.019, 
            'minRAIN_SISPI': 0.0, 
            'maxRAIN_SISPI': 0.0, 
            'minT2': 301.632, 
            'minRAIN_GPM': 0.0
        }

        values = ea.find_min_max_value('test/data/')
        self.assertEquals(v, values['values']) 
        




if __name__ == '__main__':
    unittest.main(testRunner=HTMLTestRunner(
        output='/home/maibyssl/Ariel/rain/SisPIRainfall/test/results',
        report_name='prepare_data_unittest_results'))

