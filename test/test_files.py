import os
import unittest
import numpy as np
from files.netcdf import NetCDF

"""
How implement unit test for void methods?
"""

class TestNetCDF(unittest.TestCase):
    netcdf_file = NetCDF('wrfout_d03_2017-07-09_23:00:00.nc')

    def test_xlon(self):
        real_xlong = np.loadtxt('XLONG.csv', delimiter=',')
        self.assertEqual(self.netcdf_file.xlon, real_xlong[0])

    @unittest.skip('Not implemented yet')
    def test_xlat(self):
        real_xlat = np.loadtxt('XLAT.csv', delimiter=',')
        self.assertEqual(self.netcdf_file.xlon, real_xlat[:, 0])

    @unittest.skip('Not implemented yet')
    def test_vars(self):
        pass


if __name__ == '__main__':
    unittest.main()

