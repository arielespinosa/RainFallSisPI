from rna.evaluate_models import *
from preprocess.file import read_serialize_file

try:
    from rna.plot_rna_results import predict
except:
    pass


def eval_all_models_grid_predictions():
    pass


import os

if __name__ == "__main__":
  
    #model_name = "tts_south_mlp_model_1"
    #history = read_serialize_file("rna/outputs/{0}/history_{0}.dat".format(model_name))
    #min_max_history_statistics(history)
    
    #model_statistics_new("rna/outputs")
    #best_models_by_type_and_point("rna/outputs", 'rna/best_models_by_type_and_point.txt')



    #models_grid_statistics_to_excel("rna/grid_predictions", 'rna/grid_predictions/statistics_grid_2.csv')
    #find_best_model_statistics_new("rna/outputs", 'rna/best_models.txt')
    #statistics_to_excel_new("rna/grid_predictions", 'rna/grid_predictions/statistics_grid.csv')

    #predict(2017, 6, 4, 20, "/home/maibyssl/Ariel/rain/SisPIRainfall/rna")
    #eval_models_grid_predictions("2017060423")
    #models_grid_statistics_to_excel("rna/grid_predictions", 'rna/grid_predictions/statistics_grid_output.csv')

    pass

     
