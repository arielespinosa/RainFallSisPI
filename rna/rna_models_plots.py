import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *
from preprocess.file import *
import numpy as np
from pylab import *
import os
from preprocess.file import read_serialize_file, fileslist
#from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase


def plot_interpolation():  
    
    sispi  = np.array(read_serialize_file("outputs/sispi_points.dat"))
    cmorph = np.array(read_serialize_file("outputs/cmorph_points.dat"))

    plt.plot(sispi[:, 0], sispi[:, 1])
    plt.set_xlabel='lon'
    plt.set_ylabel='lat'
    plt.axis = [sispi[:, 0], sispi[:, 1]]
    plt.set_title('Interpolation')    
    plt.ion()
    plt.show()
    
def plot_history(model_name):

    history = read_serialize_file("rna/outputs/{0}/history_{0}.dat".format(model_name))

    suptitle('Métricas del modelo ' + model_name)

    fig, ax = plt.subplots(3, 1, figsize=(15,15))

    # Plot training & validation loss values
    ax[0].set_title('Pérdida')  
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_ylabel('Perdida')
    ax[0].set_xlabel('Época')
    ax[0].legend(['Entrenamiento', 'Prueba'], loc='best')
 
    # Plot training & validation mse values
    ax[1].set_title('Error cuadrático medio')  
    ax[1].plot(history.history['mse'])
    ax[1].plot(history.history['val_mse'])
    ax[1].set_ylabel('Error cuadrático medio')
    ax[1].set_xlabel('Época')
    ax[1].legend(['Entrenamiento', 'Prueba'], loc='best')

    # Plot training & validation mae values
    ax[2].set_title('Error absoluto')  
    ax[2].plot(history.history['mae'])
    ax[2].plot(history.history['val_mae'])
    ax[2].set_ylabel('Error absoluto')
    ax[2].set_xlabel('Época')
    ax[2].legend(['Entrenamiento', 'Prueba'], loc='best')

    plt.subplots_adjust(hspace=0.5)
   
    fig.savefig("rna/outputs/{}/training_statistics.png".format(model_name))
    plt.close(fig)

def plot_metrics():

    i = 1

    while i < 11:
        model = "rna/models2/svm_model_" + str(i) + ".dat"
        name = model.split("/")[-1].split(".")[0]
        history = read_serialize_file(model)
        plot_history(history, name)
        i += 1

        del history

"""
def plot_sispi_vs_rna_hourly_statistics():
    file = os.path.join(RNA_RESULTS, "v2_svm_statistics_sispi_rna_hourly_scaled.dat")
    statistics = read_serialize_file(file)
  
    for model in statistics.keys():
        suptitle('Análisis del modelo ' + model)

        mean = []
        std  = []

        for hour in statistics[model].keys():
            mean.append([statistics[model][hour]["RS"]["mean"], 
                         statistics[model][hour]["RN"]["mean"],
                         statistics[model][hour]["RC"]["mean"]])

            std.append([statistics[model][hour]["RS"]["std"],
                        statistics[model][hour]["RN"]["std"],
                        statistics[model][hour]["RC"]["std"]])

        mean = np.array(mean)
        std  = np.array(std)

        # Plot training & validation accuracy values
        subplot(2,2,1)
        title('Media')  
        plot(mean[:, 0])
        plot(mean[:, 1])
        #plot(mean[:, 2])
        set_ylabel('Media')
        set_xlabel('Hora')
        legend(['SISPI', 'RNA', 'CMORPH'], loc='best')

        # Plot training & validation loss values
        subplot(2,2,2)
        title('Desv. estándar')  
        plot(std[:, 0])
        plot(std[:, 1])
        #plot(std[:, 2])
        set_ylabel('Desv. estándar')
        set_xlabel('Hora')
        legend(['SISPI', 'RNA', 'CMORPH'], loc='best')
    
        plt.show()
"""

def plot_predictions():
    min_lat = 130
    max_lat = 145
    min_lon = 105
    max_lon = 130
    # Loading scalers
    x_scaler = read_serialize_file("rna/outputs/mlp_model_0/scaler_x_mlp_model_0.dat")
    y_scaler = read_serialize_file("rna/outputs/mlp_model_0/scaler_y_mlp_model_0.dat")

    predictions_files = read_serialize_file("rna/predict_files_list.dat")

    predictions_results = read_serialize_file("rna/outputs/mlp_model_0/predictions_mlp_model_0.dat")
    preditcions_results = y_scaler.inverse_transform(predictions_results)
    
    i = 0
    for file in predictions_files:
        data = read_serialize_file(file)

        rain_gpm   = data["RAIN_GPM"][min_lat:max_lat, min_lon:max_lon].reshape(375, )
        rain_sispi = data["RAIN_SISPI"][min_lat:max_lat, min_lon:max_lon].reshape(375, )
        rain_rna = preditcions_results[i]
        

        #plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
    
        suptitle('Predicciones del modelo para el día')

        # Plot training & validation accuracy values
        #subplot(2,2,1)
        #title('Media')  

        #plt.scatter(rain_gpm, label = "GPM", marker="o", facecolor='blue')
        #plt.scatter(rain_sispi, label = "SisPI", marker="o", facecolor='green')
        #plt.scatter(rain_rna, label = "RNA", marker="o", facecolor='red')

        plot(rain_gpm, "ro")
        plot(rain_sispi, "bo")
        plot(rain_rna, "go")

        #plot(mean[:, 2]))
        #legend(['CMORPH', 'RNA', 'SISPI'], loc='best')

        plt.show()

"""
def plot_predictions_mean(model_name = None):
    prediction_files = fileslist("rna/outputs/predictions/", name_condition=model_name)

    for file in prediction_files:
        file_name = file.split("/")[-1].split(".")[0]    
        statistics = read_serialize_file(file)
              

        x_scaler_name = "rna/scalers/x_" + file_name + ".dat"
        y_scaler_name = "rna/scalers/y_" + file_name + ".dat"
        x_scaler = read_serialize_file(x_scaler_name)
        y_scaler = read_serialize_file(y_scaler_name)

        cmorph = y_scaler.inverse_transform(statistics["cmorph"])
        rna    = y_scaler.inverse_transform(statistics["rna"])
        sispi  = y_scaler.inverse_transform(statistics["sispi"])
   
        #print(sispi.shape)
        #print(cmorph.shape)
        #print(rna.shape)

        #return 0
        
        suptitle('Análisis del modelo ' + file_name)

        mean = []
        std  = []


        for i in range(len(sispi)):
            #mean.append({"cmorph_mean":np.mean(cmorph[i]), "sispi_mean":np.mean(sispi[i]), "rna_mean":np.mean(rna[i]) }) 
            mean.append([np.mean(cmorph[i]), np.mean(sispi[i]), np.mean(rna[i])])         
            std.append([np.std(cmorph[i]), np.std(sispi[i]), np.std(rna[i])])

        mean = np.array(mean)
        std  = np.array(std)

        # Plot training & validation accuracy values
        subplot(2,2,1)
        title('Media')  
        plot(mean[:, 0])
        plot(mean[:, 1])
        plot(mean[:, 2])
        set_ylabel('Media')
        set_xlabel('Instancias')
        legend(['CMORPH', 'SISPI', 'RNA'], loc='best')

        # Plot training & validation loss values
        subplot(2,2,2)
        title('Desv. estándar')  
        plot(std[:, 0])
        plot(std[:, 1])
        plot(std[:, 2])
        set_ylabel('Desv. estándar')
        set_xlabel('Instancias')
        legend(['CMORPH', 'SISPI', 'RNA'], loc='best')

        plt.show()
"""

def plot_sispi():

        
        #precip = np.loadtxt("data/dataset2_2017121101.txt", delimiter=',')

        #lat = np.loadtxt("data/sispi_lat.txt", delimiter=',')
       
        #lon = np.loadtxt("data/sispi_lon.txt", delimiter=',')

        return 0
     

        fig = plt.figure(figsize=[10,8])
        map = Basemap(llcrnrlon=np.min(lon), llcrnrlat=np.min(lat),\
        urcrnrlon=np.max(lon), urcrnrlat=np.max(lat),\
        projection='merc', resolution='i')
        clevs = [0,0.1,2,5,10,15,20,30,40,50,70,100,150,200,250]    
        x,y = map(lon,lat)

        map.drawcoastlines()
        map.drawcountries()

        map.drawmeridians(np.arange(np.int(np.min(lon)),np.int(np.max(lon)),3), labels=[0,0,0,1])
        map.drawparallels(np.arange(np.int(np.min(lat)),np.int(np.max(lat)),3), labels=[1,0,0,0])
        cs = map.contourf(x,y,precip,clevs,cmap=cm.s3pcpn)
        cs = map.colorbar(cs,location='bottom',pad="10%")
        cs.set_label('mm/h')
        plt.savefig('data/prueba/precipitacion1.png')
        plt.close(fig)

def plot_scatter_results(model_name):

        sispi = np.loadtxt("rna/outputs/{}/sispi.txt".format(model_name), delimiter=',')
        gpm   = np.loadtxt("rna/outputs/{}/gpm.txt".format(model_name), delimiter=',')
        rna   = np.loadtxt("rna/outputs/{}/rna.txt".format(model_name), delimiter=',')
        
        s = len(rna)
        t = np.arange(0, s)

        sispi = sispi.reshape(s, 1)
        gpm   = gpm.reshape(s, 1)

        fig, ax = plt.subplots(2, 1, figsize=(15,15))

        ax[0].scatter(t, sispi, label='SisPI')
        ax[0].scatter(t, gpm, label='GPM')
        ax[0].set_ylabel('mm3')
        ax[0].set_xlabel('tiempo t')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].scatter(t, rna, label='RNA')
        ax[1].scatter(t, gpm, label='GPM')
        ax[1].set_ylabel('mm3')
        ax[1].grid(True)
        ax[1].legend()

        fig.savefig("rna/outputs/{}/scatter_plot.png".format(model_name))
        plt.close(fig)

def plot_histogram_results(model_name, subplots=True):
    sispi = np.loadtxt("rna/outputs/{}/sispi.txt".format(model_name), delimiter=',')
    gpm = np.loadtxt("rna/outputs/{}/gpm.txt".format(model_name), delimiter=',')
    rna = np.loadtxt("rna/outputs/{}/rna.txt".format(model_name), delimiter=',')

    s = len(rna)
    t = np.arange(0, s)

    sispi = sispi.reshape(s, 1)
    gpm = gpm.reshape(s, 1)

    if subplots:
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))

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

        fig.savefig("rna/outputs/{}/histogram_plot.png".format(model_name))
    else:
        fig, ax = plt.subplots(figsize=(15, 15))

        ax.plot(t, rna, label='RNA')
        ax.plot(t, gpm, label='GPM')
        ax.plot(t, sispi, label='SisPI')
        ax.set_ylabel('mm3')
        ax.grid(True)
        ax.legend()

        fig.savefig("rna/outputs/{}/histogram_plot_2.png".format(model_name))

    plt.close(fig)

#plot_sispi()
#plot_predictions_mean(model_name = "v0_mlp")
#plot_sispi_vs_rna_hourly_statistics()
#plot_statistics()
#plot_metrics()






