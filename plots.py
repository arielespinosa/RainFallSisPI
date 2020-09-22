import os
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from pylab import *
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import cm

#from files.netcdf import NetCDF
#from preprocess.files import read_serialize_file


import threading
from multiprocessing import Process

# Delete from here
import time
from datetime import datetime, timedelta                  
from os import path, listdir, scandir
from os.path import abspath
import shutil


def plot_precipitation_map(precipitation, ylat, xlon):
    clevs = [0,0.1,2,5,10,15,20,30,40,50,70,100,150,200,250]

    map = Basemap(llcrnrlon=np.min(xlon), llcrnrlat=np.min(ylat),\
    urcrnrlon=np.max(xlon), urcrnrlat=np.max(ylat),\
    projection='merc', resolution='i')
     
    x,y = map(xlon, ylat)

    map.drawcoastlines()
    map.drawcountries()
    map.drawstates(color="b")
    map.drawmeridians(np.arange(np.int(np.min(xlon)),np.int(np.max(xlon)),1), labels=[0,0,0,1])
    map.drawparallels(np.arange(np.int(np.min(ylat)),np.int(np.max(ylat)),1), labels=[1,0,0,0])

    cs = map.contourf(x,y,precipitation,clevs,cmap=cm.s3pcpn)
    cs = map.colorbar(cs,location='bottom',pad="20%")
    cs.set_label('mm/h')

def plot_gpm_precipitation(rain):    
    lat  = np.loadtxt("data/prueba2/gpm_m_lat.txt",  delimiter=',')
    lon  = np.loadtxt("data/prueba2/gpm_m_lon.txt",  delimiter=',')    
    plot_precipitation_map(rain, lat, lon)

def plot_sispi_precipitation(rain):
    ylat = np.loadtxt("data/sispi_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/sispi_lon.txt", delimiter=',')
    plot_precipitation_map(rain, ylat, xlon)

def plot_sispi_habana_precipitation(rain):
    ylat = np.loadtxt("data/habana_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/habana_lon.txt", delimiter=',')

    plot_precipitation_map(rain, ylat, xlon)

# Plot interpolated GPM data rainfall over Havana SisPI grid
def plot_gpm_interpolated_habana_precipitation(rain):
    ylat = np.loadtxt("data/habana_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/habana_lon.txt", delimiter=',')
    plot_precipitation_map(rain, ylat, xlon)

# Plot none interpolated GPM data rainfall over Havana grid
def plot_gpm_habana_precipitation(rain):
    ylat = np.loadtxt("data/gpm_habana_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/gpm_habana_lon.txt", delimiter=',')
    plot_precipitation_map(rain, ylat, xlon)

def plot_rna_habana_precipitation(rain):
    ylat = np.loadtxt("data/habana_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/habana_lon.txt", delimiter=',')

    plot_precipitation_map(rain, ylat, xlon)


# Distinct plot map options for controler function
def plot_gpm_sispi_precipitation_map(data, from_txt):
    plt.subplot(121)
    plot_gpm_precipitation(data["gpm"])
    plt.title('GPM')
    plt.subplot(122)
    plot_sispi_precipitation(data["sispi"], from_txt=from_txt)
    plt.title('SisPI')

def plot_gpm_sispi_rna_map(data):
    plt.subplot(2,3,1)
    plot_dataset_gpm_habana_precipitation(data["dataset_file"])
    plt.title('GPM')
    plt.subplot(2,3,2)
    plot_dataset_sispi_habana_precipitation(data["dataset_file"])
    plt.title('SisPI')
    plt.subplot(2,3,3)
    plot_dataset_rna_habana_precipitation(data["prediction_file"], data["i"])
    plt.title('RNA')

def plot_gpm_sispi_rna_map2(data):
    plt.subplot(2,3,1)
    plot_gpm_habana_precipitation(data["rain_gpm_interpolated"])
    plt.title('GPM')
    plt.subplot(2,3,2)
    plot_sispi_habana_precipitation(data["rain_sispi"])
    plt.title('SisPI')
    plt.subplot(2,3,3)
    plot_rna_habana_precipitation(data["rain_rna"])
    plt.title('RNA')    

def plot_gpm_sispi_rna_map3(data):
    plt.subplot(2,2,1)
    plot_gpm_interpolated_habana_precipitation(data["rain_gpm_interpolated"])
    plt.title('GPM Interpolado')
    plt.subplot(2,2,2)
    plot_gpm_habana_precipitation(data["rain_gpm"])
    plt.title('GPM')
    plt.subplot(2,2,3)
    plot_sispi_habana_precipitation(data["rain_sispi"])
    plt.title('SisPI')
    plt.subplot(2,2,4)
    plot_rna_habana_precipitation(data["rain_rna"])
    plt.title('RNA')  

def plot_gpm_interpolated_sispi_precipitation_map(file):
    ylat = np.loadtxt("data/sispi_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/sispi_lon.txt", delimiter=',')

    dataset = NetCDF(file)
    rain_sispi = dataset.dataset["RAIN_SISPI"]
    rain_gpm = dataset.dataset["RAIN_GPM"]

    plt.subplot(121)
    plot_precipitation_map(rain_sispi, ylat, xlon)
    plt.title('RAIN_SISPI')

    plt.subplot(122)
    plot_precipitation_map(rain_gpm, ylat, xlon)
    plt.title('RAIN_GPM_INTERPOLATED')

def plot(file, model_type, folder):
    filename = file.split("/")[-1]

    dataset = read_serialize_file(file)

    ylat = np.loadtxt("data/habana_lat.txt", delimiter=',')
    xlon = np.loadtxt("data/habana_lon.txt", delimiter=',')

    plt.suptitle('Relación entre GPM, SisPI y la RNA para las {3} h del dia {2}/{1}/{0}'.format(filename[2:6], filename[6:8], filename[8:10], filename[10:12]))

    # Plot SisPI
    subplot(2,2,1)
    rain = dataset["RAIN_SISPI"][130:145, 105:130]
    plot_precipitation_map(rain, ylat, xlon)
    plt.title('SisPI')    

    # Plot GPM
    subplot(2,2,2)
    rain = dataset["RAIN_GPM"][130:145, 105:130]
    plot_precipitation_map(rain, ylat, xlon)
    plt.title('GPM') 
   
    # Plot RNA
    subplot(2,2,3)
    rain = read_serialize_file("/home/maibyssl/Ariel/rain/ariel/rna/outputs/{0}/predictions/{1}".format(model_type, filename))
    plot_precipitation_map(rain, ylat, xlon)
    plt.title('RNA') 

    #plt.show()
    plt.savefig(os.path.join(folder, '{}.png'.format(filename[2:12])))
    #plt.close(fig)

def controler(data, map, from_txt = False, subtitle=None, show=False, save_filename=None):    
    #fig = plt.figure(1, (10, 8))
    fig, ax = plt.subplots(1, figsize=(15, 10))

    if map == "both":
        plot_gpm_sispi_precipitation_map(data, from_txt=from_txt)
    elif map == "sispi":
        plot_sispi_precipitation(data["rain_sispi"], default=default, from_txt=from_txt)
    elif map == "gpm":
        plot_gpm_precipitation(data["rain_gpm"])
    elif map == "dataset_interpolated":
        plot_gpm_interpolated_sispi_precipitation_map(data)
    elif map == "predictions":
        plot_gpm_sispi_rna_map(data)
    elif map == "predictions2":
        plot_gpm_sispi_rna_map2(data)
    elif map == "predictions3":
        plot_gpm_sispi_rna_map3(data)

    if subtitle:
        plt.suptitle(subtitle)

    if show:
        plt.show()
    if save_filename:
        plt.savefig(save_filename)
        
def plot_for_period(start_time, end_time, path):
    s_time = start_time

    while s_time < end_time:
        time = "%04d%02d%02d%02d" % (s_time.year, s_time.month, s_time.day, s_time.hour)

        data = {
            "sispi": "data/sispi_hourly_dataset/d_{}.dat".format(time),
            "gpm": "data/gpm_dataset/gpm_hourly_{}.txt".format(time),
        }

        controler(data=data, map="both", default=True, 
        subtitle='SisPI and GPM relationship for %04d-%02d-%02d-%02d' % (s_time.year, s_time.month, s_time.day, s_time.hour),
        show=False, 
        save_filename=os.path.join(path, time))

        s_time += timedelta(hours=1)

def plot_results_2(data, save_filename, subtitule):

    sispi = data["rain_sispi"]
    gpm   = data["rain_gpm"]
    rna   = data["rain_rna"]
    
    s = 375
    t = np.arange(0, s)

    sispi = sispi.reshape(s, 1)
    gpm   = gpm.reshape(s, 1)
    rna   = rna.reshape(s, 1)

    fig, ax = plt.subplots(3, 1, figsize=(15,15))

    ax[0].plot(t, rna, label='RNA')
    ax[0].set_ylabel('mm3')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t, sispi, label='SisPI')
    ax[1].set_ylabel('mm3')
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(t, gpm, label='GPM')
    ax[2].set_ylabel('mm3')
    ax[2].grid(True)
    ax[2].legend()

    plt.suptitle(subtitle)

    fig.savefig(save_filename)
    plt.close(fig)

def plot_results(data, save_filename, subtitule):

    sispi = data["rain_sispi"]
    gpm   = data["rain_gpm_interpolated"]
    rna   = data["rain_rna"]

    s = 375
    t = np.arange(0, s)

    sispi = sispi.reshape(s, 1)
    gpm   = gpm.reshape(s, 1)
    rna   = rna.reshape(s, 1)

    fig, ax = plt.subplots(figsize=(15,15))

    ax.plot(t, rna, label='RNA')
    ax.plot(t, gpm, label='GPM Interpolado')
    ax.plot(t, sispi, label='SisPI')
    ax.set_ylabel('mm3')
    ax.grid(True)
    ax.legend()

    plt.suptitle(subtitle)

    fig.savefig(save_filename)
    plt.close(fig)


class Plot:

    # Ok no borrar
    @classmethod
    def grid_prediction(cls, y, m, d, h, save_path):
        fig, ax = plt.subplots(figsize=(20, 20))
        fig.patch.set_facecolor('#ffffff')
        dataset = np.loadtxt("data/predict_grid/d_{:4d}{:02d}{:02d}{:02d}.txt".format(y, m, d, h), delimiter=',')

        ylat = np.loadtxt("data/habana_lat.txt", delimiter=',')
        xlon = np.loadtxt("data/habana_lon.txt", delimiter=',')

        #fig = plt.figure()
        #plt.style.use('seaborn')
        plt.suptitle('Relación entre GPM, SisPI y la RNA para las {:02d} UTC del {:02d}/{:02d}/{:4d}'.format(h, d, m, y))

        # Plot GPM
        fig.add_subplot(2,3,1)
        rain = dataset[:, -1].reshape(15, 25)
        rain[rain < 1] = 0
        plot_precipitation_map(rain, ylat, xlon)
        plt.title('GPM') 

        # Plot SisPI
        fig.add_subplot(2,3,2)
        rain = dataset[:, -2].reshape(15, 25)
        rain[rain < 1] = 0
        plot_precipitation_map(rain, ylat, xlon)
        plt.title('SisPI')    

        # Plot RNA
        fig.add_subplot(2,3,3)
        rain = np.loadtxt("rna/grid_predictions/d_{:4d}{:02d}{:02d}{:02d}.txt".format(y, m, d, h), delimiter=',')
        rain[rain < 1] = 0
        plot_precipitation_map(rain, ylat, xlon)
        plt.title('RNA') 

        plt.show()

    @classmethod
    def plot_grid_prediction_monthly(cls, y, m, d, h, save_path):        
        t = datetime(y, m, d, h)

        while t.month == m:        
            p = Process(target=cls.grid_prediction, args=(t.year, t.month, t.day, t.hour, save_path))
            p.start() 
            p.join()
            
            t += timedelta(hours=1)
     

if __name__ == "__main__":
    Plot.grid_prediction(2017, 6, 1, 22, "save_path")

    """
    fig = plt.figure()

    fig.patch.set_facecolor('#ffffff')
    fig.patch.set_alpha(0.7)

    ax = fig.add_subplot(111)

    t = arange(0.0, 2.0, 0.01)
    s = sin(2*pi*t)
    #plot(t, s)

    ax.plot(t, s) #plot(range(10))

    ax.patch.set_facecolor('#ffffff')
    ax.patch.set_alpha(0.5)

    # If we don't specify the edgecolor and facecolor for the figure when
    # saving with savefig, it will override the value we set earlier!
    #fig.savefig('exemple_02.png', facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()
    """



