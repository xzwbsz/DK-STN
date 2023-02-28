#导入相关包
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


#数据标准化
def mean_std(data_xxx):
    data_mean=np.mean(data_xxx)
    data_std=np.std(data_xxx)
    final_data=data_xxx-data_mean
    final_data=np.true_divide(final_data,data_std)
    return final_data

#减去120天平均值
def avg_120(test_data,time_length):
    new_data=np.zeros(test_data[120:time_length,:,:].shape)
    for i in range(120,time_length):
        mean_120=np.mean(test_data[i-120:i,:,:],axis=0)
        new_data[i-120,:,:]=test_data[i,:,:]-mean_120[:,:]
    final_data=np.array(new_data)
    return final_data


def babj_create_dataset():

    #根据文件名中的日期打开相对应的三个文件
    path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/babj/OLR/'
    path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/babj/U200/'
    path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/babj/U850/'
    path_sst='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/babj/SST/'


    path_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/s2s_RMM/babj/'

    path_re_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/re_RMM/babj/'

    babj_X=[]
    babj_Y=[]
    re_babj_Y = []

            #判断是否包含闰年的2.29号
    run_year=np.array([200801,200802,200803,200804,200805,200806,
                       201201,201202,201203,201204,201205,201206,
                       201601,201602,201603,201604,201605,201606,
                      202001,202002,202003,202004,202005,202006])
    run_year_1=np.array([200801,200802,
                         201201,201202,
                         201601,201602,
                         202001,202002])
    run_year_2=np.array([200803,200804,200805,200806,
                         201203,201204,201205,201206,
                         201603,201604,201605,201606,
                      202003,202004,202005,202006])

    #获取文件名
    filenames = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/babj_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        str_filename = str(int(filenames[i]))

        babj_olr = netCDF4.Dataset(path_olr+str_filename+'_add_120redata_babj_olr.nc')
        babj_sst = netCDF4.Dataset(path_sst+str_filename+'_babj_sst.nc')
        babj_u200 = netCDF4.Dataset(path_u200+str_filename+'_add_120redata_babj_u200.nc')
        babj_u850 = netCDF4.Dataset(path_u850+str_filename+'_add_120redata_babj_u850.nc')

        #获得s2s的babj的rmm
        babj_rmm = np.load(path_rmm + str_filename + '_babj_rmm.npy')
        babj_re_rmm = np.load(path_re_rmm + str_filename + '_babj_rmm.npy')

        date_time=int(str_filename)
        print(date_time)

        data_babj_olr = np.array(babj_olr.variables['olr'])
        data_babj_sst = np.array(babj_sst.variables['sst'])
        data_babj_u200 = np.array(babj_u200.variables['u200'])
        data_babj_u850 = np.array(babj_u850.variables['u850'])



        #日期数字化
        date_yearmonth = date_time // 100
            #如果在闰年并且包含2.29，就使用闰年的气候态

        if date_yearmonth in run_year:
        #读取气候态
            sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/OLR/'
            sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/U200/'
            sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/U850/'
            sea_olr=np.load(sea_path1 + str_filename[4:8]+'_run_seasonal_cycle_olr.npy')
            sea_u200=np.load(sea_path2 + str_filename[4:8]+'_run_seasonal_cycle_u200.npy')
            sea_u850=np.load(sea_path3 + str_filename[4:8]+'_run_seasonal_cycle_u850.npy')
            #sea_olr=np.true_divide(sea_olr,3600)
        else:
        #读取气候态
            sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/OLR/'
            sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/U200/'
            sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/babj/U850/'
            sea_olr=np.load(sea_path1 + str_filename[4:8]+'_seasonal_cycle_olr.npy')
            sea_u200=np.load(sea_path2 + str_filename[4:8]+'_seasonal_cycle_u200.npy')
            sea_u850=np.load(sea_path3 + str_filename[4:8]+'_seasonal_cycle_u850.npy')

        #减去气候态
        data_olr = data_babj_olr - sea_olr
        data_u200 = data_babj_u200 - sea_u200
        data_u850 = data_babj_u850 - sea_u850
        data_olr = - data_olr

        #减去前120天的平均
        #数据长度
        time_length = len(data_olr)
        reduec_120data_olr = avg_120(data_olr,time_length)
        reduec_120data_u200 = avg_120(data_u200,time_length)
        reduec_120data_u850 = avg_120(data_u850,time_length)
        print(reduec_120data_olr.shape,reduec_120data_u200.shape,reduec_120data_u850.shape)

        #最后判断截取
        if date_yearmonth in run_year_1:
            reduec_120data_olr = reduec_120data_olr[0:60]
            reduec_120data_u200 = reduec_120data_u200[0:60]
            reduec_120data_u850 = reduec_120data_u850[0:60]

        elif date_yearmonth in run_year_2:
            reduec_120data_olr = reduec_120data_olr[1:61]
            reduec_120data_u200 = reduec_120data_u200[1:61]
            reduec_120data_u850 = reduec_120data_u850[1:61]

        else:
            reduec_120data_olr = reduec_120data_olr
            reduec_120data_u200 = reduec_120data_u200
            reduec_120data_u850 = reduec_120data_u850
        #数据标准化
        mean_olr = mean_std(reduec_120data_olr)
        mean_sst = mean_std(data_babj_sst)
        mean_u200 = mean_std(reduec_120data_u200)
        mean_u850 = mean_std(reduec_120data_u850)


        #升维度，再合并
        u200_data = np.expand_dims(mean_u200,axis=3)
        u850_data = np.expand_dims(mean_u850,axis=3)
        olr_data = np.expand_dims(mean_olr,axis=3)
        sst_data = np.expand_dims(mean_sst,axis=3)

        #合并数组（60，13，144，4）
        data_combine=np.concatenate((olr_data, u850_data, u200_data, sst_data),axis=3)
        #print('combine data:',data_combine.shape)

        for j in range(0, 15, 7):
            babj_X.append(data_combine[j:j+7,:,:,:])
            babj_Y.append(babj_rmm[j+7:j+42,:])
            re_babj_Y.append(babj_re_rmm[j+7:j+42, :])

    babj_X = np.array(babj_X)
    babj_Y = np.array(babj_Y)
    re_babj_Y = np.array(re_babj_Y)

    return babj_X, babj_Y, re_babj_Y


def ecmf_create_dataset():

    path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/OLR/'
    path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U200/'
    path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U850/'
    path_sst = '/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/SST/'

    path_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/s2s_RMM/ecmf/'
    path_re_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/re_RMM/ecmf/'

    ecmf_X=[]
    ecmf_Y=[]
    re_ecmf_Y = []

    #判断是否包含闰年的2.29号
    run_year=np.array([200402,200403,200404,200405,200406,
                       200802,200803,200804,200805,200806,
                       201202,201203,201204,201205,201206,
                       201602,201603,201604,201605,201606,
                      202002,202003,202004,202005,202006])
    run_year_1=np.array([200402,200802,201202,201602,202002])
    run_year_2=np.array([200403,200404,200405,200406,
                         200803,200804,200805,200806,
                         201203,201204,201205,201206,
                         201603,201604,201605,201606,
                      202003,202004,202005,202006])
    run_year_3=np.array([20040121,20080121,20120121,20160121,20200121])

    #获取文件名
    filenames=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/ecmf_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        str_filename = str(int(filenames[i]))

        babj_olr = netCDF4.Dataset(path_olr+str_filename+'_add_120redata_ecmf_olr.nc')
        babj_sst = netCDF4.Dataset(path_sst+str_filename+'_ecmf_sst.nc')
        babj_u200 = netCDF4.Dataset(path_u200+str_filename+'_add_120redata_ecmf_u200.nc')
        babj_u850 = netCDF4.Dataset(path_u850+str_filename+'_add_120redata_ecmf_u850.nc')

        #获得s2s的babj的rmm
        ecmf_rmm = np.load(path_rmm + str_filename + '_ecmf_rmm.npy')
        ecmf_re_rmm = np.load(path_re_rmm + str_filename + '_ecmf_rmm.npy')

        date_time=int(str_filename)
        print(date_time)

        data_babj_olr = np.array(babj_olr.variables['olr'])
        data_babj_sst = np.array(babj_sst.variables['sst'])
        data_babj_u200 = np.array(babj_u200.variables['u200'])
        data_babj_u850 = np.array(babj_u850.variables['u850'])

        #日期数字化
        date_yearmonth = date_time // 100



        #如果在闰年并且包含2.29，就使用闰年的气候态
        if (date_yearmonth in run_year)|(date_time in run_year_3):
        #读取气候态
            sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/OLR/'
            sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U200/'
            sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'
            sea_olr=np.load(sea_path1+str_filename[4:8]+'_run_seasonal_cycle_olr.npy')
            sea_u200=np.load(sea_path2+str_filename[4:8]+'_run_seasonal_cycle_u200.npy')
            sea_u850=np.load(sea_path3+str_filename[4:8]+'_run_seasonal_cycle_u850.npy')
            #sea_olr=np.true_divide(sea_olr,3600)
        else:
        #读取气候态
            sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/OLR/'
            sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U200/'
            sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'
            sea_olr=np.load(sea_path1+str_filename[4:8]+'_seasonal_cycle_olr.npy')
            sea_u200=np.load(sea_path2+str_filename[4:8]+'_seasonal_cycle_u200.npy')
            sea_u850=np.load(sea_path3+str_filename[4:8]+'_seasonal_cycle_u850.npy')
            #sea_olr=np.true_divide(sea_olr,3600)

        #减去气候态
        data_olr = data_babj_olr - sea_olr
        data_u200 = data_babj_u200 - sea_u200
        data_u850 = data_babj_u850 - sea_u850
        data_olr = - data_olr

        #减去前120天的平均
        #数据长度
        time_length = len(data_olr)
        reduec_120data_olr = avg_120(data_olr,time_length)
        reduec_120data_u200 = avg_120(data_u200,time_length)
        reduec_120data_u850 = avg_120(data_u850,time_length)
        print(reduec_120data_olr.shape,reduec_120data_u200.shape,reduec_120data_u850.shape)

        #最后判断截取
        if (date_yearmonth in run_year_1)|(date_time in run_year_3):
            reduec_120data_olr = reduec_120data_olr[0:46]
            reduec_120data_u200 = reduec_120data_u200[0:46]
            reduec_120data_u850 = reduec_120data_u850[0:46]
            print("situation 1")
        elif date_yearmonth in run_year_2:
            reduec_120data_olr = reduec_120data_olr[1:47]
            reduec_120data_u200 = reduec_120data_u200[1:47]
            reduec_120data_u850 = reduec_120data_u850[1:47]
            print("situation 2")
        else:
            reduec_120data_olr = reduec_120data_olr
            reduec_120data_u200 = reduec_120data_u200
            reduec_120data_u850 = reduec_120data_u850
            
        #数据标准化
        mean_olr = mean_std(reduec_120data_olr)
        mean_sst = mean_std(data_babj_sst)
        mean_u200 = mean_std(reduec_120data_u200)
        mean_u850 = mean_std(reduec_120data_u850)


        #升维度，再合并
        u200_data = np.expand_dims(mean_u200,axis=3)
        u850_data = np.expand_dims(mean_u850,axis=3)
        olr_data = np.expand_dims(mean_olr,axis=3)
        sst_data = np.expand_dims(mean_sst,axis=3)

        #合并数组（60，13，144，4）
        data_combine=np.concatenate((olr_data, u850_data, u200_data, sst_data),axis=3)
        #print('combine data:',data_combine.shape)


        for j in range(0, 4, 1):
            ecmf_X.append(data_combine[j:j+7,:,:,:])
            ecmf_Y.append(ecmf_rmm[j+7:j+42,:])
            re_ecmf_Y.append(ecmf_re_rmm[j+7:j+42, :])

    ecmf_X = np.array(ecmf_X)
    ecmf_Y = np.array(ecmf_Y)
    re_ecmf_Y = np.array(re_ecmf_Y)
    return ecmf_X, ecmf_Y, re_ecmf_Y

def re_babj_create_dataset():

    #根据文件名中的日期打开相对应的三个文件
    path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/OLR/'
    path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/U200/'
    path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/U850/'
    path_sst = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/babj/SST/'


    path_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/s2s_RMM/babj/'

    path_re_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/re_RMM/babj/'

    babj_X=[]
    babj_Y=[]
    re_babj_Y = []


    #获取文件名
    filenames = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/babj_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        str_filename = str(int(filenames[i]))

        data_babj_olr = np.load(path_olr+str_filename+'_babj_olr.npy')
        data_babj_sst = np.load(path_sst+str_filename+'_babj_sst.npy')
        data_babj_u200 = np.load(path_u200+str_filename+'_babj_u200.npy')
        data_babj_u850 = np.load(path_u850+str_filename+'_babj_u850.npy')

        data_babj_sst = data_babj_sst[120:, :, :]

        #获得s2s的babj的rmm
        babj_rmm = np.load(path_rmm + str_filename + '_babj_rmm.npy')
        babj_re_rmm = np.load(path_re_rmm + str_filename + '_babj_rmm.npy')

        date_time=int(str_filename)
        print(date_time)



        #读取气候态
        sea_path1 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/OLR/'
        sea_path2 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/U200/'
        sea_path3 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/babj/U850/'

        sea_olr = np.load(sea_path1 + str_filename +'_babj_seasonal_olr.npy')
        sea_u200 = np.load(sea_path2 + str_filename +'_babj_seasonal_u200.npy')
        sea_u850 = np.load(sea_path3 + str_filename +'_babj_seasonal_u850.npy')



        #减去气候态
        data_olr = data_babj_olr - sea_olr
        data_u200 = data_babj_u200 - sea_u200
        data_u850 = data_babj_u850 - sea_u850
        data_olr = - data_olr

        #减去前120天的平均
        #数据长度
        time_length = len(data_olr)
        reduec_120data_olr = avg_120(data_olr,time_length)
        reduec_120data_u200 = avg_120(data_u200,time_length)
        reduec_120data_u850 = avg_120(data_u850,time_length)
        print(reduec_120data_olr.shape,reduec_120data_u200.shape,reduec_120data_u850.shape)

        #数据标准化
        mean_olr = mean_std(reduec_120data_olr)
        mean_sst = mean_std(data_babj_sst)
        mean_u200 = mean_std(reduec_120data_u200)
        mean_u850 = mean_std(reduec_120data_u850)


        #升维度，再合并
        u200_data = np.expand_dims(mean_u200,axis=3)
        u850_data = np.expand_dims(mean_u850,axis=3)
        olr_data = np.expand_dims(mean_olr,axis=3)
        sst_data = np.expand_dims(mean_sst,axis=3)

        #合并数组（60，13，144，4）
        data_combine=np.concatenate((olr_data, u850_data, u200_data, sst_data),axis=3)
        #print('combine data:',data_combine.shape)

        for j in range(0, 15, 7):
            babj_X.append(data_combine[j:j+7,:,:,:])
            babj_Y.append(babj_rmm[j+7:j+42,:])
            re_babj_Y.append(babj_re_rmm[j+7:j+42, :])

    babj_X = np.array(babj_X)
    babj_Y = np.array(babj_Y)
    re_babj_Y = np.array(re_babj_Y)

    return babj_X, babj_Y, re_babj_Y


def re_ecmf_create_dataset():

    path_olr = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/OLR/'
    path_u200 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/U200/'
    path_u850 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/U850/'
    path_sst = '/WdHeDisk/users/zhangnong/MJO/711_test/re_data/ecmf/SST/'

    path_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/s2s_RMM/ecmf/'
    path_re_rmm = '/WdHeDisk/users/zhangnong/MJO/711_test/re_RMM/ecmf/'

    ecmf_X=[]
    ecmf_Y=[]
    re_ecmf_Y = []


    #获取文件名
    filenames=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/ecmf_date_sort.txt')
    print('------')
    print(filenames[0])
    print(len(filenames))
    for i in range(0, len(filenames) - 9):

        str_filename = str(int(filenames[i]))

        data_babj_olr = np.load(path_olr+str_filename+'_ecmf_olr.npy')
        data_babj_sst = np.load(path_sst+str_filename+'_ecmf_sst.npy')
        data_babj_u200 = np.load(path_u200+str_filename+'_ecmf_u200.npy')
        data_babj_u850 = np.load(path_u850+str_filename+'_ecmf_u850.npy')

        data_babj_sst = data_babj_sst[120:, :, :]

        #获得s2s的babj的rmm
        ecmf_rmm = np.load(path_rmm + str_filename + '_ecmf_rmm.npy')
        ecmf_re_rmm = np.load(path_re_rmm + str_filename + '_ecmf_rmm.npy')

        date_time=int(str_filename)
        print(date_time)


        #读取气候态
        sea_path1 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/OLR/'
        sea_path2 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/U200/'
        sea_path3 = '/WdHeDisk/users/zhangnong/MJO/711_test/re_seasonal_cycle/ecmf/U850/'

        sea_olr = np.load(sea_path1+str_filename +'_ecmf_seasonal_olr.npy')
        sea_u200 = np.load(sea_path2+str_filename +'_ecmf_seasonal_u200.npy')
        sea_u850 = np.load(sea_path3+str_filename +'_ecmf_seasonal_u850.npy')

        #减去气候态
        data_olr = data_babj_olr - sea_olr
        data_u200 = data_babj_u200 - sea_u200
        data_u850 = data_babj_u850 - sea_u850
        data_olr = - data_olr

        #减去前120天的平均
        #数据长度
        time_length = len(data_olr)
        reduec_120data_olr = avg_120(data_olr,time_length)
        reduec_120data_u200 = avg_120(data_u200,time_length)
        reduec_120data_u850 = avg_120(data_u850,time_length)
        print(reduec_120data_olr.shape,reduec_120data_u200.shape,reduec_120data_u850.shape)


        #数据标准化
        mean_olr = mean_std(reduec_120data_olr)
        mean_sst = mean_std(data_babj_sst)
        mean_u200 = mean_std(reduec_120data_u200)
        mean_u850 = mean_std(reduec_120data_u850)


        #升维度，再合并
        u200_data = np.expand_dims(mean_u200,axis=3)
        u850_data = np.expand_dims(mean_u850,axis=3)
        olr_data = np.expand_dims(mean_olr,axis=3)
        sst_data = np.expand_dims(mean_sst,axis=3)

        #合并数组（60，13，144，4）
        data_combine=np.concatenate((olr_data, u850_data, u200_data, sst_data),axis=3)
        #print('combine data:',data_combine.shape)


        for j in range(0, 4, 1):
            ecmf_X.append(data_combine[j:j+7,:,:,:])
            ecmf_Y.append(ecmf_rmm[j+7:j+42,:])
            re_ecmf_Y.append(ecmf_re_rmm[j+7:j+42, :])

    ecmf_X = np.array(ecmf_X)
    ecmf_Y = np.array(ecmf_Y)
    re_ecmf_Y = np.array(re_ecmf_Y)
    return ecmf_X, ecmf_Y, re_ecmf_Y


babj_X, babj_Y ,re_babj_Y = babj_create_dataset()
print(babj_X.shape, babj_Y.shape, re_babj_Y.shape)

ecmf_X, ecmf_Y, re_ecmf_Y = ecmf_create_dataset()

r_babj_X, r_s2s_babj_Y, r_re_babj_Y = re_babj_create_dataset()

r_ecmf_X, r_s2s_ecmf_Y, r_re_ecmf_Y = re_ecmf_create_dataset()

print(babj_X.shape, babj_Y.shape, re_babj_Y.shape)
print(ecmf_X.shape, ecmf_Y.shape, re_ecmf_Y.shape)
print(r_babj_X.shape, r_s2s_babj_Y.shape, r_re_babj_Y.shape)
print(r_re_ecmf_Y.shape, r_s2s_ecmf_Y.shape, r_re_ecmf_Y.shape)

t_X = r_ecmf_X[5000: ]
t_Y = r_s2s_ecmf_Y[5000: ]
re_Y = r_re_ecmf_Y[5000: ]

#测试集400
test_sample=[i for i in range(len(t_X))]
random.shuffle(test_sample)

test_X = t_X[test_sample[0:400]]
test_Y = t_Y[test_sample[0:400]]
re_test_Y = re_Y[test_sample[0:400]]

#训练集2100，验证集700
train_sample=[i for i in range(4900)]
random.shuffle(train_sample)

t_X = np.concatenate((babj_X[train_sample[0:4800]], ecmf_X[train_sample[0:4800]], r_babj_X[train_sample[0:4800]], r_ecmf_X[train_sample[0:4800]]), axis=0)
t_Y = np.concatenate((babj_Y[train_sample[0:4800]], ecmf_Y[train_sample[0:4800]], r_s2s_babj_Y[train_sample[0:4800]], r_s2s_ecmf_Y[train_sample[0:4800]]), axis=0)
re_t_Y = np.concatenate((re_babj_Y[train_sample[0:4800]], re_ecmf_Y[train_sample[0:4800]], r_re_babj_Y[train_sample[0:4800]], r_re_ecmf_Y[train_sample[0:4800]]), axis=0)

train_X = t_X[0:18800]
train_Y = t_Y[0:18800]
re_train_Y = re_t_Y[0:18800]

valid_X = t_X[18800:]
valid_Y = t_Y[18800:]
re_valid_Y = re_t_Y[18800:]

print(train_X.shape, train_Y.shape, re_train_Y.shape)
print(valid_X.shape, valid_Y.shape, re_valid_Y.shape)
print(test_X.shape, test_Y.shape, re_test_Y.shape)


np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/X_train_for7_7_35_sample.npy', train_X)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/Y_train_for7_7_35_sample.npy', train_Y)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/re_Y_train_for7_7_35_sample.npy', re_train_Y)

np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/X_valid_for7_7_35_sample.npy', valid_X)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/Y_valid_for7_7_35_sample.npy', valid_Y)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/re_Y_valid_for7_7_35_sample.npy', re_valid_Y)

np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/X_test_for7_7_35_sample.npy', test_X)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/Y_test_for7_7_35_sample.npy', test_Y)
np.save('/WdHeDisk/users/zhangnong/MJO/908_test/data/mixed_dataset_for7_7_35/re_Y_test_for7_7_35_sample.npy', re_test_Y)


