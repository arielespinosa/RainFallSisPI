from rna.grid_predictions import predict_month, predict



if __name__ == "__main__":

    predict(2017, 1, 1, 8)
    #p1 = Process(target=predict_month, args=(2017, 1, 1))
    #p1.start()

    print("Finish!")