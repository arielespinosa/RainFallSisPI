from .keras_models import MLP
from preprocess.file import read_serialize_file
import numpy as np
import os
from datetime import datetime, timedelta
import time
from multiprocessing import Process


# Prepare data (time steps = 5) for make predictions on the Habana space grid
def prepare_data(year, month, day, hour):
    dataset = []

    for i in range(5):
        file = "data/full_dataset_txt/d_%04d%02d%02d%02d.txt" % (year, month, day, hour - i)
        data = np.loadtxt(file, delimiter=",")
        dataset.insert(0, data[:, 2])

    data = np.stack(np.array(dataset), axis=1)
    np.savetxt("data/predict_grid/d_%04d%02d%02d%02d.txt" % (year, month, day, hour), data, fmt="%7.3f", delimiter=',')


def predict_area(data, area):
    predictions = []

    if area is "north":
        x_scaler = read_serialize_file("rna/outputs/tts_5_north_mlp_model_7/x_scaler.dat")
        y_scaler = read_serialize_file("rna/outputs/tts_5_north_mlp_model_7/y_scaler.dat")
        points = x_scaler.transform(data[:125])

        model = MLP("rna/outputs/tts_5_north_mlp_model_7/tts_5_north_mlp_model_7.h5")

    elif area is "east":
        x_scaler = read_serialize_file("rna/outputs/tts_5_east_mlp_model_9/x_scaler.dat")
        y_scaler = read_serialize_file("rna/outputs/tts_5_east_mlp_model_9/y_scaler.dat")

        points = [data[i:i + 8] for i in range(125, 250, 25)]
        points = np.array(points)
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
        points = x_scaler.transform(points)

        model = MLP("rna/outputs/tts_5_east_mlp_model_9/tts_5_east_mlp_model_9.h5")

    elif area is "center":
        x_scaler = read_serialize_file("rna/outputs/tts_5_center_mlp_model_25/x_scaler.dat")
        y_scaler = read_serialize_file("rna/outputs/tts_5_center_mlp_model_25/y_scaler.dat")

        points = [data[i:i + 9] for i in range(132, 257, 25)]
        points = np.array(points)
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
        points = x_scaler.transform(points)

        model = MLP("rna/outputs/tts_5_center_mlp_model_25/tts_5_center_mlp_model_25.h5")

    elif area is "west":
        x_scaler = read_serialize_file("rna/outputs/tts_5_west_mlp_model_19/x_scaler.dat")
        y_scaler = read_serialize_file("rna/outputs/tts_5_west_mlp_model_19/y_scaler.dat")

        points = [data[i:i + 8] for i in range(142, 267, 25)]
        points = np.array(points)
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
        points = x_scaler.transform(points)

        model = MLP("rna/outputs/tts_5_west_mlp_model_19/tts_5_west_mlp_model_19.h5")

    elif area is "south":
        x_scaler = read_serialize_file("rna/outputs/tts_5_south_mlp_model_29/x_scaler.dat")
        y_scaler = read_serialize_file("rna/outputs/tts_5_south_mlp_model_29/y_scaler.dat")
        points = x_scaler.transform(data[250:])

        model = MLP("rna/outputs/tts_5_south_mlp_model_29/tts_5_south_mlp_model_29.h5")

    results = model.predict(points)
    results = y_scaler.inverse_transform(results)

    del model
    return results


def unify_predict_area(north, east, center, west, south):
    n = north.reshape(5, 25)
    e = east.reshape(5, 8)
    c = center.reshape(5, 9)
    w = west.reshape(5, 8)
    s = south.reshape(5, 25)

    middle = np.hstack((e, c, w))
    return np.vstack((n, middle, s))


def predict(y, m, d, h):
    prepare_data(y, m, d, h)
    file = "d_%04d%02d%02d%02d.txt" % (y, m, d, h)

    try:
        data = np.loadtxt("data/predict_grid/{}".format(file), delimiter=",")

        n = predict_area(data, "north")
        e = predict_area(data, "east")
        c = predict_area(data, "center")
        w = predict_area(data, "west")
        s = predict_area(data, "south")

        grid = unify_predict_area(n, e, c, w, s)

        np.savetxt("rna/grid_predictions/{}".format(file), grid, fmt="%7.3f", delimiter=",")
    except:
        pass


def predict_month(y, m, d):
    t = datetime(y, m, d)

    while t.month == m:
        p = Process(target=predict, args=(t.year, t.month, t.day, t.hour))
        p.start()
        p.join()

        t += timedelta(hours=1)
