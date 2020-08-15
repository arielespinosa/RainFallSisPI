import os
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Process
from scipy.interpolate import NearestNDInterpolator
from preprocess.prepare_netcdf import PrepareNetCDF as pncdf
from preprocess.prepare_gpm import gpm_to_hourly, export_gpm_lat_lon
from preprocess.file import fileslist, read_serialize_file, write_serialize_file
from settings import configuration as config


class PrepareData:

    @classmethod
    def prepare_sispi(cls):
        print("Descomprimiendo los ficheros tar.gz y serializando los ficheros NetCDF...")
        # Ver como hacer para que no se ejecute lo siguiente hasta que no termine este proceso.
        # Como los hilos se ejecutan no en el principal, el programa continua.
        pncdf.start_serialization()
        print("Eliminando los ficheros NetCDF correspondientes a las 25-36 horas...")
        pncdf.remove_not_required_files()
        print("Copiando los ficheros a la dirección correcta...")
        pncdf.put_sispi_files_on_right_path()
        print("Exportando latitud y longitud de SisPI...")
        pncdf.export_sispi_lat_lon()

    @classmethod
    def prepare_gpm(cls, grid=None, gpm_dir=None):
        if grid is None:
            grid = {"max_lat": 24.35, "min_lat": 19.25, "max_lon": -73.75, "min_lon": -85.75}

        gpm_files = fileslist(config['DIRS']['GPM_DIR']) if gpm_dir else fileslist(config['DIRS']['GPM_DIR'])
        gpm_files.sort()
        gpm_to_hourly(grid, gpm_files)
        export_gpm_lat_lon(config['GPM_FILE'], grid)

    @classmethod
    def __interpolate_gpm_to_sispi(cls, month, gpm_lat, gpm_lon, sispi_lat, sispi_lon):
        gpm_points = np.concatenate((gpm_lon.reshape(gpm_lon.size, 1), gpm_lat.reshape(gpm_lat.size, 1)), axis=1)
        sispi_points = np.concatenate((sispi_lon.reshape(sispi_lon.size, 1), sispi_lat.reshape(sispi_lat.size, 1)), axis=1)

        t = datetime(2017, month, 1)
        while t.month <= month:    
            try:        
                gpm_data = np.loadtxt(
                    os.path.join(config['DIRS']['GPM_OUTPUT_DIR'],
                                'd_%04d%02d%02d%02d.txt' % (t.year, t.month, t.day, t.hour)),
                                delimiter=',')

                interpolator = NearestNDInterpolator(gpm_points, gpm_data.flatten())
                interp_values = interpolator(sispi_points).reshape(183, 411)

                np.savetxt(
                    os.path.join(config['DIRS']['GPM_INTERPOLATED'],
                                'd_%04d%02d%02d%02d.txt' % (t.year, t.month, t.day, t.hour)),
                    interp_values, delimiter=",", fmt="%7.2f")

                t += timedelta(hours=1)
                del interpolator

            except:
                pass
    
    @classmethod
    def __interpolate_gpm(cls, month_start=1, month_end=12):
        gpm_lat = np.loadtxt(config['GPM_LAT'], delimiter=',')
        gpm_lon = np.loadtxt(config['GPM_LON'], delimiter=',')
        sispi_lat = np.loadtxt(config['SISPI_LAT'], delimiter=',')
        sispi_lon = np.loadtxt(config['SISPI_LON'], delimiter=',')

        gpm_lon, gpm_lat = np.meshgrid(gpm_lon, gpm_lat)
        sispi_lon, sispi_lat = np.meshgrid(sispi_lon, sispi_lat)

        process_list = [Process(target=cls.__interpolate_gpm_to_sispi, args=(month, gpm_lat, gpm_lon, sispi_lat, sispi_lon))
                        for month in range(month_start, month_end+1)]

        i, c = 0, 0
        for process in process_list:
            if c > 2:
                process_list[i-3].join()
                process_list[i-2].join()
                process_list[i-1].join()
                c = 0
            process.start()

            c += 1
            i += 1

    @classmethod
    def __join_sispi_and_gpm(cls):
        sispi_files = fileslist(config['DIRS']['SISPI_SERIALIZED_OUTPUT_DIR'])

        for file in sispi_files:
            filename = file.split('/')[-1].split('.')[0]
            gpm_file = os.path.join(config['DIRS']['GPM_INTERPOLATED'], filename + '.txt')

            sispi = read_serialize_file(file)
            gpm = np.loadtxt(gpm_file, delimiter=",")

            # Replacing RAINNC or RAINC var for RAIN_SISPI
            sispi.update({'RAIN_SISPI': sispi['RAINC']})
            sispi.pop('RAINC')

            # Adding GPM to dataset
            sispi.update({'RAIN_GPM': gpm})

            write_serialize_file(sispi, os.path.join(config['DIRS']['DATASET'], filename + '.dat'))

    @classmethod
    def combine_sispi_and_gpm(cls, interpoated=True):
        if not interpoated:
            cls.__interpolate_gpm()
        cls.__join_sispi_and_gpm()

    @classmethod
    def set_dataset_habana(cls, min_lat=130, max_lat=145, min_lon=105, max_lon=130):
        dataset = fileslist(config['DIRS']['DATASET'])

        for file in dataset:
            data = read_serialize_file(file)
            value = {
                'T2': data['T2'][min_lat:max_lat, min_lon:max_lon],
                'Q2': data['Q2'][min_lat:max_lat, min_lon:max_lon],
                'RAIN_SISPI': data['RAIN_SISPI'][min_lat:max_lat, min_lon:max_lon],
                'RAIN_GPM': data['RAIN_GPM'][min_lat:max_lat, min_lon:max_lon],
            }
            write_serialize_file(value, os.path.join(config['DIRS']['DATASET_HABANA'], file.split('/')[-1]))

    @classmethod
    def prepare_train_dataset(cls, fname, series_len=6, point=(0, 0), dataset_is_complete=True):
        dataset, series = [], []
        first_line = True
        count = 0

        #dataset_habana = fileslist("/home/maibyssl/Ariel/rain/ariel/data/full_dataset")
        dataset_habana = fileslist(config['DIRS']['DATASET_HABANA'])
        dataset_habana.sort()

        for file in dataset_habana:
            data = read_serialize_file(file)

            if dataset_is_complete:
                # If dataset have all days-hours its not necesary read each file
                if first_line:
                    for i in range(series_len):
                        aux_data = read_serialize_file(dataset_habana[count - i])
                        value = aux_data['RAIN_SISPI'][point[0], point[1]]
                        series.insert(0, value)
                    first_line = False
                else:
                    series = dataset[count - 1][1:series_len]
                    series.append(data['RAIN_SISPI'][point[0], point[1]])

                series.append(data['RAIN_GPM'][point[0], point[1]])
                dataset.append(series)
                count += 1

            else:
                # Preguntar a Maybis por la implementacion que hice si la incluyo o no
                pass

        if fname:
            np.savetxt(fname, dataset, delimiter=',', fmt='%7.2f')

        return dataset

