import os
import unittest
import numpy as np
from files.netcdf import NetCDF
from files.gpm import GPM

"""
How implement unit test for void methods? R/ Mocks
"""

class TestNetCDF(unittest.TestCase):
    netcdf_file = NetCDF('test/wrfout_d03_2017-07-09_23:00:00.nc')

    @unittest.skip("Ready")
    def test_xlon(self):
        real_xlong = np.loadtxt('test/XLONG.csv', dtype='float32', delimiter=',')
        real_xlong = real_xlong[0].tolist()
        method_xlong = self.netcdf_file.xlon.tolist()
        self.assertListEqual(real_xlong, method_xlong)

    @unittest.skip("Ready")
    def test_xlat(self):
        real_xlat = np.loadtxt('test/XLAT.csv', dtype='float32', delimiter=',')
        real_xlat = real_xlat[:, 0].tolist()
        method_xlat = self.netcdf_file.xlat.tolist()
        self.assertListEqual(real_xlat, method_xlat)

    @unittest.skip("Take to much time. Try to do it with mocks lib")
    def test_vars(self):
        real_rainc = np.loadtxt('test/RAINC.csv', dtype='float32', delimiter=',').tolist()
        real_rainnc = np.loadtxt('test/RAINNC.csv', dtype='float32', delimiter=',').tolist()
        real_q2 = np.loadtxt('test/Q2.csv', dtype='float32', delimiter=',').tolist()
        real_t2 = np.loadtxt('test/T2.csv', dtype='float32', delimiter=',').tolist()

        data = self.netcdf_file.vars(['T2', 'Q2', 'RAINC', 'RAINNC'])

        method_rainc = data['RAINC'].tolist()
        method_rainnc = data['RAINNC'].tolist()
        method_q2 = data['Q2'].tolist()
        method_t2 = data['T2'].tolist()

        self.assertListEqual(real_rainc, method_rainc)
        self.assertListEqual(real_rainnc, method_rainnc)
        self.assertListEqual(real_t2, method_t2)
        self.assertListEqual(real_q2, method_q2)


class TestGPM(unittest.TestCase):
    grid = {"max_lat": 24.35, "min_lat": 19.25, "max_lon": -73.75, "min_lon": -85.75}
    gpm_file = GPM('test/3B-HHR.MS.MRG.3IMERG.20170101-S000000-E002959.0000.V06B.HDF5')

    @unittest.skip("Ready")
    def test_xlon(self):
        real_xlong = np.loadtxt('test/GPM_XLONG.csv', dtype='float32', delimiter=',')
        real_xlong = real_xlong.tolist()
        method_xlong = self.gpm_file.xlon.tolist()
        self.assertListEqual(real_xlong, method_xlong)

    @unittest.skip("Ready")
    def test_xlat(self):
        real_xlat = np.loadtxt('test/GPM_XLAT.csv', dtype='float32', delimiter=',')
        real_xlat = real_xlat.tolist()
        method_xlat = self.gpm_file.xlat.tolist()
        self.assertListEqual(real_xlat, method_xlat)

    @unittest.skip("Take to much time. Try to do it with mocks lib")
    def test_rain(self):
        real_rain = np.loadtxt('test/GPM_RAIN.csv', dtype='float32', delimiter=',')
        real_rain = real_rain.tolist()
        method_rain = self.gpm_file.rain(False).tolist()
        self.assertListEqual(real_rain, method_rain)


if __name__ == '__main__':
    unittest.main()

