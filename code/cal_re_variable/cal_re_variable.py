#导入相关包
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import random
import netCDF4
import datetime
#import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
#plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
#%matplotlib inline

#读取数据
data1 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u200_1950-2022.nc')
data2 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u850_1950-2022.nc')
data3 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/olr_1950-2022.nc')

data_sst_1950_2022 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/sst_1950-2022.nc')

data_u200 = np.array(data1.variables['u200'][120:,:,:])
data_u850 = np.array(data2.variables['u850'][120:,:,:])
data_olr = np.array(data3.variables['olr'][120:,:,:])
data_sst = np.array(data_sst_1950_2022.variables['sst'][120:,:,:])

print(data_u200.shape, data_u850.shape, data_olr.shape, data_sst.shape)
#读取气候态
seasonal_cycle_u200 = np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u200_sample.npy')
seasonal_cycle_u850 = np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u850_sample.npy')
seasonal_cycle_olr = np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_olr_sample.npy')

seasonal_cycle_u200 = seasonal_cycle_u200[120: ,:, :]
seasonal_cycle_u850 = seasonal_cycle_u850[120:, :, :]
seasonal_cycle_olr = seasonal_cycle_olr[120:, :, :]

print(seasonal_cycle_olr.shape, seasonal_cycle_u200.shape, seasonal_cycle_u850.shape)

re_data_rmm=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/pca_RMM_1950_2022.txt')
re_data_time=np.array(re_data_rmm[:,0])
print(re_data_time.shape)

#判断babj得到的数据是否正确
def true_babj(data):

    length = data.shape[0]
    lon = data.shape[1]
    lat = data.shape[2]

    if(length == 180) & (lon == 13) & (lat == 144):
        return 0
    else:
        return 1

def true_ecmf(data):

    length = data.shape[0]
    lon = data.shape[1]
    lat = data.shape[2]

    if(length == 166) & (lon == 13) & (lat == 144):
        return 0
    else:
        return 1

#构造babj模式的olr等变量
#需要找到对应filename位置，然后读取前120天+后60天
def babj_re_create():
    #获取文件名
    filenames = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/babj_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        #得到文件名的整数模式,即时间开始
        date_time = int(filenames[i])

        str_filename = str(date_time)

        #根据文件名的时间定位到olr等变量的位置
        #定位时间
        location = np.argwhere(re_data_time == date_time)
        print("location:", int(location))

        #加上前120天
        start_location = int(location) - 120
        fina_location = int(location) + 60

        #提取olr
        babj_olr = data_olr[start_location: fina_location, :, :]

        babj_seasonal_olr = seasonal_cycle_olr[start_location: fina_location, :, :]

        #提取u200
        babj_u200 = data_u200[start_location: fina_location, :, :]

        babj_seasonal_u200 = seasonal_cycle_u200[start_location: fina_location, :, :]

        #提取u850
        babj_u850 = data_u850[start_location: fina_location, :, :]

        babj_seasonal_u850 = seasonal_cycle_u850[start_location: fina_location, :, :]

        #提取sst
        babj_sst = data_sst[start_location: fina_location, :, :]

        #存储数据
        #保存文件
        save_path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/OLR/'
        save_path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/U200/'
        save_path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/U850/'
        save_path_sst = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/SST/'

        save_path_olr_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/OLR/'
        save_path_u200_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/U200/'
        save_path_u850_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/U850/'

        #判断数据大小是否正确
        if true_babj(babj_olr) == 0:
            np.save(save_path_olr + str_filename + '_babj_olr.npy', babj_olr)
        else:
            print('error!!!')
            return

        if true_babj(babj_u200) == 0:
            np.save(save_path_u200 + str_filename + '_babj_u200.npy', babj_u200)
        else:
            print('error!!!')
            return
        
        if true_babj(babj_u850) == 0:
            np.save(save_path_u850 + str_filename + '_babj_u850.npy', babj_u850)
        else:
            print('error!!!')
            return
        
        if true_babj(babj_sst) == 0:
            np.save(save_path_sst + str_filename + '_babj_sst.npy', babj_sst)
        else:
            print('error!!!')
            return

        if true_babj(babj_seasonal_olr) == 0:
            np.save(save_path_olr_se + str_filename + '_babj_seasonal_olr.npy', babj_seasonal_olr)
        else:
            print('error!!!')
            return

        if true_babj(babj_seasonal_u200) == 0:
            np.save(save_path_u200_se + str_filename + '_babj_seasonal_u200.npy', babj_seasonal_u200)
        else:
            print('error!!!')
            return
        
        if true_babj(babj_seasonal_u850) == 0:
            np.save(save_path_u850_se + str_filename + '_babj_seasonal_u850.npy', babj_seasonal_u850)
        else:
            print('error!!!')
            return
        
    print('success!!!')
    return


def ecmf_re_create():
    #获取文件名
    filenames = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/ecmf_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        #得到文件名的整数模式,即时间开始
        date_time = int(filenames[i])

        str_filename = str(date_time)

        #根据文件名的时间定位到olr等变量的位置
        #定位时间
        location = np.argwhere(re_data_time == date_time)
        print("location:", int(location))

        #加上前120天
        start_location = int(location) - 120
        fina_location = int(location) + 46

        #提取olr
        ecmf_olr = data_olr[start_location: fina_location, :, :]

        ecmf_seasonal_olr = seasonal_cycle_olr[start_location: fina_location, :, :]

        #提取u200
        ecmf_u200 = data_u200[start_location: fina_location, :, :]

        ecmf_seasonal_u200 = seasonal_cycle_u200[start_location: fina_location, :, :]

        #提取u850
        ecmf_u850 = data_u850[start_location: fina_location, :, :]

        ecmf_seasonal_u850 = seasonal_cycle_u850[start_location: fina_location, :, :]

        #提取sst
        ecmf_sst = data_sst[start_location: fina_location, :, :]

        #存储数据
        #保存文件
        save_path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/OLR/'
        save_path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/U200/'
        save_path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/U850/'
        save_path_sst = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/SST/'

        save_path_olr_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/OLR/'
        save_path_u200_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/U200/'
        save_path_u850_se = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/U850/'

        #判断数据大小是否正确
        if true_ecmf(ecmf_olr) == 0:
            np.save(save_path_olr + str_filename + '_ecmf_olr.npy', ecmf_olr)
        else:
            print('error!!!')
            return

        if true_ecmf(ecmf_u200) == 0:
            np.save(save_path_u200 + str_filename + '_ecmf_u200.npy', ecmf_u200)
        else:
            print('error!!!')
            return
        
        if true_ecmf(ecmf_u850) == 0:
            np.save(save_path_u850 + str_filename + '_ecmf_u850.npy', ecmf_u850)
        else:
            print('error!!!')
            return
        
        if true_ecmf(ecmf_sst) == 0:
            np.save(save_path_sst + str_filename + '_ecmf_sst.npy', ecmf_sst)
        else:
            print('error!!!')
            return

        if true_ecmf(ecmf_seasonal_olr) == 0:
            np.save(save_path_olr_se + str_filename + '_ecmf_seasonal_olr.npy', ecmf_seasonal_olr)
        else:
            print('error!!!')
            return

        if true_ecmf(ecmf_seasonal_u200) == 0:
            np.save(save_path_u200_se + str_filename + '_ecmf_seasonal_u200.npy', ecmf_seasonal_u200)
        else:
            print('error!!!')
            return
        
        if true_ecmf(ecmf_seasonal_u850) == 0:
            np.save(save_path_u850_se + str_filename + '_ecmf_seasonal_u850.npy', ecmf_seasonal_u850)
        else:
            print('error!!!')
            return
        
    print('success!!!')
    return      

babj_re_create()
sleep(10)
ecmf_re_create()

        

        

