import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import (explained_variance_score,
                            mean_absolute_error, 
                            mean_squared_error, 
                            mean_squared_log_error, 
                            median_absolute_error,
                            r2_score)

def min_max_history_statistics(history):

    # Training & validation loss values
    print("\nTrain loss value: Max: %7.4f, Lower: %7.4f" % (np.max(history.history['loss']), np.min(history.history['loss'])))
    print("Eval  loss value: Max: %7.4f, Lower: %7.4f\n" % (np.max(history.history['val_loss']), np.min(history.history['val_loss'])))

    # Training & validation mse values
    print("\nTrain mse value: Max: %7.4f, Lower: %7.4f" % (np.max(history.history['mse']), np.min(history.history['mse']))) 
    print("Train mae value: Max: %7.4f, Lower: %7.4f\n" % (np.max(history.history['mae']), np.min(history.history['mae']))) 
    
    print("Eval  mse value: Max: %7.4f, Lower: %7.4f" % (np.max(history.history['val_mse']), np.min(history.history['val_mse']))) 
    print("Eval  mae value: Max: %7.4f, Lower: %7.4f\n" % (np.max(history.history['val_mae']), np.min(history.history['val_mae']))) 

def best_models_by_type_and_point(directory, file):
    best_e_mlp, best_e_nsvm, best_e_conv, best_e_lstm  = "","","",""
    best_n_mlp, best_n_nsvm, best_n_conv, best_n_lstm  = "","","",""
    best_s_mlp, best_s_nsvm, best_s_conv, best_s_lstm  = "","","",""
    best_w_mlp, best_w_nsvm, best_w_conv, best_w_lstm = "","","",""
    best_c_mlp, best_c_nsvm, best_c_conv, best_c_lstm = "","","",""

    best_score_e_mlp, best_score_e_nsvm, best_score_e_conv, best_score_e_lstm = 0.0, 0.0, 0.0, 0.0
    best_score_n_mlp, best_score_n_nsvm, best_score_n_conv, best_score_n_lstm = 0.0, 0.0, 0.0, 0.0
    best_score_s_mlp, best_score_s_nsvm, best_score_s_conv, best_score_s_lstm = 0.0, 0.0, 0.0, 0.0
    best_score_w_mlp, best_score_w_nsvm, best_score_w_conv, best_score_w_lstm = 0.0, 0.0, 0.0, 0.0
    best_score_c_mlp, best_score_c_nsvm, best_score_c_conv, best_score_c_lstm = 0.0, 0.0, 0.0, 0.0

    for folder in os.scandir(directory):
        path = folder.path
        folder = path 
        
        with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
            statisticians = json.load(json_file)

        if folder.find("east")>-1:                    
            if folder.find("mlp")>-1:
                if statisticians["evs_rna_gpm"] > best_score_e_mlp:
                    best_score_e_mlp = statisticians["evs_rna_gpm"]
                    best_e_mlp = path
            elif folder.find("nsvm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_e_nsvm:
                    best_score_e_nsvm = statisticians["evs_rna_gpm"]
                    best_e_nsvm = path
            elif folder.find("conv")>-1:
                if statisticians["evs_rna_gpm"] > best_score_e_conv:
                    best_score_e_conv = statisticians["evs_rna_gpm"]
                    best_e_conv = path
            elif folder.find("lstm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_e_lstm:
                    best_score_e_lstm = statisticians["evs_rna_gpm"]
                    best_e_lstm = path

        elif folder.find("north")>-1:
            if folder.find("mlp")>-1:
                if statisticians["evs_rna_gpm"] > best_score_n_mlp:
                    best_score_n_mlp = statisticians["evs_rna_gpm"]
                    best_n_mlp = path
            elif folder.find("nsvm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_n_nsvm:
                    best_score_n_nsvm = statisticians["evs_rna_gpm"]
                    best_n_nsvm = path
            elif folder.find("conv")>-1:
                if statisticians["evs_rna_gpm"] > best_score_n_conv:
                    best_score_n_conv = statisticians["evs_rna_gpm"]
                    best_n_conv = path
            elif folder.find("lstm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_n_lstm:
                    best_score_n_lstm = statisticians["evs_rna_gpm"]
                    best_n_lstm = path
            
        elif folder.find("south")>-1:
            if folder.find("mlp")>-1:
                if statisticians["evs_rna_gpm"] > best_score_s_mlp:
                    best_score_s_mlp = statisticians["evs_rna_gpm"]
                    best_s_mlp = path
            elif folder.find("nsvm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_s_nsvm:
                    best_score_s_nsvm = statisticians["evs_rna_gpm"]
                    best_s_nsvm = path
            elif folder.find("conv")>-1:
                if statisticians["evs_rna_gpm"] > best_score_s_conv:
                    best_score_s_conv = statisticians["evs_rna_gpm"]
                    best_s_conv = path
            elif folder.find("lstm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_s_lstm:
                    best_score_s_lstm = statisticians["evs_rna_gpm"]
                    best_s_lstm = path
            
        elif folder.find("west")>-1:
            if folder.find("mlp")>-1:
                if statisticians["evs_rna_gpm"] > best_score_w_mlp:
                    best_score_w_mlp = statisticians["evs_rna_gpm"]
                    best_w_mlp = path
            elif folder.find("nsvm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_w_nsvm:
                    best_score_w_nsvm = statisticians["evs_rna_gpm"]
                    best_w_nsvm = path
            elif folder.find("conv")>-1:
                if statisticians["evs_rna_gpm"] > best_score_w_conv:
                    best_score_w_conv = statisticians["evs_rna_gpm"]
                    best_w_conv = path
            elif folder.find("lstm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_w_lstm:
                    best_score_w_lstm = statisticians["evs_rna_gpm"]
                    best_w_lstm = path
            
        elif folder.find("center")>-1:
            if folder.find("mlp")>-1:
                if statisticians["evs_rna_gpm"] > best_score_c_mlp:
                    best_score_c_mlp = statisticians["evs_rna_gpm"]
                    best_c_mlp = path
            elif folder.find("nsvm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_c_nsvm:
                    best_score_c_nsvm = statisticians["evs_rna_gpm"]
                    best_c_nsvm = path
            elif folder.find("conv")>-1:
                if statisticians["evs_rna_gpm"] > best_score_c_conv:
                    best_score_c_conv = statisticians["evs_rna_gpm"]
                    best_c_conv = path
            elif folder.find("lstm")>-1:
                if statisticians["evs_rna_gpm"] > best_score_c_lstm:
                    best_score_c_lstm = statisticians["evs_rna_gpm"]
                    best_c_lstm = path
        
    with open(file, 'w') as txt_file:
        txt_file.write("Mejores modelos del este:\n")       
        txt_file.write(best_e_mlp+"\n")
        txt_file.write(best_e_nsvm+"\n")
        txt_file.write(best_e_conv+"\n")
        txt_file.write(best_e_lstm+"\n")

        txt_file.write("\nMejores modelos del norte:\n")       
        txt_file.write(best_n_mlp+"\n")
        txt_file.write(best_n_nsvm+"\n")
        txt_file.write(best_n_conv+"\n")
        txt_file.write(best_n_lstm+"\n")

        txt_file.write("\nMejores modelos del oeste:\n")       
        txt_file.write(best_w_mlp+"\n")
        txt_file.write(best_w_nsvm+"\n")
        txt_file.write(best_w_conv+"\n")
        txt_file.write(best_w_lstm+"\n")

        txt_file.write("\nMejores modelos del sur:\n")       
        txt_file.write(best_s_mlp+"\n")
        txt_file.write(best_s_nsvm+"\n")
        txt_file.write(best_s_conv+"\n")
        txt_file.write(best_s_lstm+"\n")

        txt_file.write("\nMejores modelos del centro:\n")       
        txt_file.write(best_c_mlp+"\n")
        txt_file.write(best_c_nsvm+"\n")
        txt_file.write(best_c_conv+"\n")
        txt_file.write(best_c_lstm+"\n")

def model_statistics(directory):

    for folder in os.scandir(directory):
        path = folder.path

        dataset = np.loadtxt(os.path.join(path, "x_predict.txt"), delimiter=",")
        rna = np.loadtxt(os.path.join(path, "predictions.txt"), delimiter=",")

        sispi = dataset[:, -2]
        gpm = dataset[:, -1]

        if any(rna<0) is False:
            statisticians = {
                "corrcoef_sispi_gpm" : list(np.corrcoef(sispi, gpm).reshape(-1, )),        # Correlation between sispi & gpm
                "corrcoef_rna_gpm"   : list(np.corrcoef(rna, gpm).reshape(-1, )),          # Correlation between sispi & gpm
                "std_rna"            : np.std(rna),                                  # Standar desviation between sispi & gpm
                "std_sispi"          : np.std(sispi),                               # Sispi standar desviation                
                "std_gpm"            : np.std(gpm),                               # gpm standar desviation
                "var_sispi"          : np.var(sispi),                                  # Varianza between sispi & gpm
                "var_rna"            : np.var(rna),                               # Sispi varianza    
                "var_gpm"            : np.var(gpm),                               # Sispi varianza              
                "cov_sispi_gpm"      : list(np.cov(sispi, gpm).reshape(-1, )),                                  # Covarianza between sispi & gpm
                "cov_rna_gpm"        : list(np.cov(rna, gpm).reshape(-1, )),                                  # Covarianza between sispi & gpm
                "ae_sispi_gpm"       : ae(sispi, gpm),     # Absolute error between sispi & gpm
                "ae_rna_gpm"         : ae(rna, gpm),     # Absolute error between sispi & gpm
                "mse_sispi_gpm"      : mse(sispi, gpm),  # Mean square error between sispi & gpm      
                "mse_rna_gpm"        : mse(rna, gpm),  # Mean square error between sispi & gpm        
            }

            # Saving statistics.
            with open(os.path.join(path, 'statistics.json'), 'w') as json_file:
                json.dump(statisticians, json_file)

def model_statistics_new(directory):

    for folder in os.scandir(directory):
        path = folder.path

        try:

            sispi = np.loadtxt(os.path.join(path, "sispi.txt"), delimiter=",")
            gpm = np.loadtxt(os.path.join(path, "gpm.txt"), delimiter=",")
            rna = np.loadtxt(os.path.join(path, "rna.txt"), delimiter=",")

            if any(rna<0) is False:
                statisticians = {
                    "corrcoef_sispi_gpm" : list(np.corrcoef(sispi, gpm).reshape(-1, )),        # Correlation between sispi & gpm
                    "corrcoef_rna_gpm"   : list(np.corrcoef(rna, gpm).reshape(-1, )),          # Correlation between sispi & gpm
                    "r2_rna_gpm"         : r2_score(gpm, rna),
                    "r2_sispi_gpm"       : r2_score(gpm, sispi),
                    "evs_rna_gpm"        : explained_variance_score(gpm, rna),  
                    "evs_sispi_gpm"      : explained_variance_score(gpm, sispi),         
                    "mae_sispi_gpm"      : mean_absolute_error(gpm, sispi),
                    "mae_rna_gpm"        : mean_absolute_error(gpm, rna), 
                    "med_ae_sispi_gpm"   : median_absolute_error(gpm, sispi),
                    "med_ae_rna_gpm"     : median_absolute_error(gpm, rna), 
                    "mse_sispi_gpm"      : mean_squared_error(gpm, sispi),     
                    "mse_rna_gpm"        : mean_squared_error(gpm, rna),  
                    "msle_sispi_gpm"     : mean_squared_log_error(gpm, sispi),     
                    "msle_rna_gpm"       : mean_squared_log_error(gpm, rna),        
                }

                # Saving statistics.
                with open(os.path.join(path, 'statistics.json'), 'w') as json_file:
                    json.dump(statisticians, json_file)
        except:
            pass

def find_best_model_statistics(directory, file):

    models = []

    for folder in os.scandir(directory):
        path = folder.path

        try:
            with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
                statisticians = json.load(json_file)

            if statisticians["mse_rna_gpm"] <= statisticians["mse_sispi_gpm"] or statisticians["ae_rna_gpm"] <= statisticians["ae_sispi_gpm"]:
                models.append(path)
        except:
            pass
   
    with open(file, 'w') as txt_file:
        for model in models:
            txt_file.write(model)
            txt_file.write("\n")

def find_best_model_statistics_new(directory, file):

    best_e, best_n, best_s, best_w, best_c = None, None, None, None, None
    best_score_e, best_score_n, best_score_s, best_score_w, best_score_c = 0.0, 0.0, 0.0, 0.0, 0.0

    for folder in os.scandir(directory):
        path = folder.path
        folder = path.split("/")[-1]
        
        try:
            with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
                statisticians = json.load(json_file)

            if folder.find("east")>-1:
                if statisticians["evs_rna_gpm"] > best_score_e:
                    best_score_e = statisticians["evs_rna_gpm"]
                    best_e = path

            elif folder.find("north")>-1:
                if statisticians["evs_rna_gpm"] > best_score_n:
                    best_score_n = statisticians["evs_rna_gpm"]
                    best_n = path

            elif folder.find("south")>-1:
                if statisticians["evs_rna_gpm"] > best_score_s:
                    best_score_s = statisticians["evs_rna_gpm"]
                    best_s = path

            elif folder.find("west")>-1:
                if statisticians["evs_rna_gpm"] > best_score_w:
                    best_score_w = statisticians["evs_rna_gpm"]
                    best_w = path

            elif folder.find("center")>-1:
                if statisticians["evs_rna_gpm"] > best_score_c:
                    best_score_c = statisticians["evs_rna_gpm"]
                    best_c = path
        except:
            pass
   
    with open(file, 'w') as txt_file:        
        txt_file.write(best_e+"\n")
        txt_file.write(best_n+"\n")
        txt_file.write(best_s+"\n")
        txt_file.write(best_w+"\n")
        txt_file.write(best_c)

def statistics_to_excel(directory, file):

    columns = ["modelo", "corrcoef_sispi_gpm", "corrcoef_rna_gpm", "std_rna", "std_sispi", "std_gpm","var_sispi", "var_rna", 
        "var_gpm", "cov_sispi_gpm", "cov_rna_gpm", "ae_sispi_gpm", "ae_rna_gpm", "mse_sispi_gpm", "mse_rna_gpm"]

    values = []

    for folder in os.scandir(directory):
        path = folder.path

        try:
            with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
                statisticians = json.load(json_file)

            data = [
                path.split("/")[-1],
                format(statisticians["corrcoef_sispi_gpm"][1], "7.6f"),
                format(statisticians["corrcoef_rna_gpm"][1], "7.6f"),
                format(statisticians["std_rna"], "7.6f"),
                format(statisticians["std_sispi"], "7.6f"),
                format(statisticians["std_gpm"], "7.6f"),
                format(statisticians["var_sispi"], "7.6f"),
                format(statisticians["var_rna"], "7.6f"),
                format(statisticians["var_gpm"], "7.6f"),
                format(statisticians["cov_sispi_gpm"][1], "7.6f"),
                format(statisticians["cov_rna_gpm"][1], "7.6f"),
                format(statisticians["ae_sispi_gpm"], "7.6f"),
                format(statisticians["ae_rna_gpm"], "7.6f"),
                format(statisticians["mse_sispi_gpm"], "7.6f"),
                format(statisticians["mse_rna_gpm"], "7.6f"),               
            ]

            values.append(data)

        except:
            pass

    df = pd.DataFrame(values, columns=columns)
    df.to_csv(file, mode='w')

def statistics_to_excel_new(directory, file):

    columns = ["modelo", "corrcoef_sispi_gpm", 
        "corrcoef_rna_gpm", "r2_rna_gpm"  , "r2_sispi_gpm",
        "evs_rna_gpm"  , "evs_sispi_gpm"     , "mae_sispi_gpm"   
        , "mae_rna_gpm" , "med_ae_sispi_gpm", "med_ae_rna_gpm", 
        "mse_sispi_gpm", "mse_rna_gpm"       , "msle_sispi_gpm"  
        , "msle_rna_gpm"]

    values = []

    for folder in os.scandir(directory):
        path = folder.path        

        with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
            statisticians = json.load(json_file)               

        data = [
            path.split("/")[-1],
            format(statisticians["corrcoef_sispi_gpm"][1], "7.6f"),
            format(statisticians["corrcoef_rna_gpm"][1], "7.6f"),
            format(statisticians["r2_rna_gpm"], "7.6f"),
            format(statisticians["r2_sispi_gpm"], "7.6f"),
            format(statisticians["evs_rna_gpm"], "7.6f"),
            format(statisticians["evs_sispi_gpm"], "7.6f"),
            format(statisticians["mae_sispi_gpm"], "7.6f"),
            format(statisticians["mae_rna_gpm"], "7.6f"),
            format(statisticians["med_ae_sispi_gpm"], "7.6f"),
            format(statisticians["med_ae_rna_gpm"], "7.6f"),
            format(statisticians["mse_sispi_gpm"], "7.6f"),
            format(statisticians["mse_rna_gpm"], "7.6f"),
            format(statisticians["msle_sispi_gpm"], "7.6f"),
            format(statisticians["msle_rna_gpm"], "7.6f"),              
        ]
        
        values.append(data)

    df = pd.DataFrame(values, columns=columns)
    df.to_csv(file, mode='w')


# Since here use only for grid predictions evaluation
def eval_models_grid_predictions(file):
    dataset = np.loadtxt("data/predict_grid/d_{}.txt".format(file), delimiter=",")
    rna = np.loadtxt("rna/grid_predictions/d_{}.txt".format(file), delimiter=",")

    gpm = dataset[:, -1],
    gpm = gpm[0]
    sispi = dataset[:, -2],
    sispi = sispi[0]
    rna = rna.reshape(-1, )

    gpm[gpm < 1] = 0
    sispi[sispi < 1] = 0
    rna[rna < 1] = 0

    statisticians = {
        "corrcoef_sispi_gpm" : list(np.corrcoef(sispi, gpm).reshape(-1, )),        # Correlation between sispi & gpm
        "corrcoef_rna_gpm"   : list(np.corrcoef(rna, gpm).reshape(-1, )),          # Correlation between sispi & gpm
        "r2_rna_gpm"         : r2_score(gpm, rna),
        "r2_sispi_gpm"       : r2_score(gpm, sispi),
        "evs_rna_gpm"        : explained_variance_score(gpm, rna),  
        "evs_sispi_gpm"      : explained_variance_score(gpm, sispi),         
        "mae_sispi_gpm"      : mean_absolute_error(gpm, sispi),
        "mae_rna_gpm"        : mean_absolute_error(gpm, rna), 
        "med_ae_sispi_gpm"   : median_absolute_error(gpm, sispi),
        "med_ae_rna_gpm"     : median_absolute_error(gpm, rna), 
        "mse_sispi_gpm"      : mean_squared_error(gpm, sispi),     
        "mse_rna_gpm"        : mean_squared_error(gpm, rna),  
        "msle_sispi_gpm"     : mean_squared_log_error(gpm, sispi),     
        "msle_rna_gpm"       : mean_squared_log_error(gpm, rna),        
    }

    # Creating folder for save statistics
    os.mkdir("rna/grid_predictions/{}".format(file))

    # Saving statistics.
    with open("rna/grid_predictions/{}/statistics.json".format(file), 'w') as json_file:
        json.dump(statisticians, json_file)

def models_grid_statistics_to_excel(directory, file):

    columns = ["modelo", "corrcoef_sispi_gpm", 
        "corrcoef_rna_gpm", "r2_rna_gpm"  , "r2_sispi_gpm",
        "evs_rna_gpm"  , "evs_sispi_gpm"     , "mae_sispi_gpm"   
        , "mae_rna_gpm" , "med_ae_sispi_gpm", "med_ae_rna_gpm", 
        "mse_sispi_gpm", "mse_rna_gpm"       , "msle_sispi_gpm"  
        , "msle_rna_gpm"]

    values = []

    for folder in os.scandir(directory):
        if folder.is_dir():
            path = folder.path

            with open(os.path.join(path, 'statistics.json'), 'r') as json_file:
                statisticians = json.load(json_file)

            data = [
                path.split("/")[-1],
                format(statisticians["corrcoef_sispi_gpm"][1], "7.6f"),
                format(statisticians["corrcoef_rna_gpm"][1], "7.6f"),
                format(statisticians["r2_rna_gpm"], "7.6f"),
                format(statisticians["r2_sispi_gpm"], "7.6f"),
                format(statisticians["evs_rna_gpm"], "7.6f"),
                format(statisticians["evs_sispi_gpm"], "7.6f"),
                format(statisticians["mae_sispi_gpm"], "7.6f"),
                format(statisticians["mae_rna_gpm"], "7.6f"),
                format(statisticians["med_ae_sispi_gpm"], "7.6f"),
                format(statisticians["med_ae_rna_gpm"], "7.6f"),
                format(statisticians["mse_sispi_gpm"], "7.6f"),
                format(statisticians["mse_rna_gpm"], "7.6f"),
                format(statisticians["msle_sispi_gpm"], "7.6f"),
                format(statisticians["msle_rna_gpm"], "7.6f"),
            ]

            values.append(data)

    df = pd.DataFrame(values, columns=columns)
    df.to_csv(file, mode='w')
