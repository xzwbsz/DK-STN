#导入相关包
from matplotlib.pyplot import axis
import numpy as np
import os
#import matplotlib.pyplot as plt
import random
import netCDF4
import datetime
import math
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
#plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
#%matplotlib inline
from sklearn.decomposition import PCA
#获取文件名
filenames=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/ecmf_date_sort.txt')
print('---------')
print(filenames[0])
print(len(filenames))
#减去120天平均值
def avg_120(test_data,time_length):
    new_data=np.zeros(test_data[120:time_length,:,:].shape)
    for i in range(120,time_length):
        mean_120=np.mean(test_data[i-120:i,:,:],axis=0)
        new_data[i-120,:,:]=test_data[i,:,:]-mean_120[:,:]
    final_data=np.array(new_data)
    return final_data
#eof,获取时间模态
def PCs(data_olr,data_u850,data_u200,lamda_vector):
    #先拼接一下(61,144)
    data_combine=np.concatenate((data_olr,data_u850,data_u200),axis=1)
    #print(data_combine.shape)
    rmm=np.dot(data_combine,lamda_vector)
    eof1=51.37508737 
    eof2=48.82820937
    lamda1=math.sqrt(eof1)
    lamda2=math.sqrt(eof2)
    rmm1=np.true_divide(rmm[:,0],lamda1)
    rmm2=np.true_divide(rmm[:,1],lamda2)
    RMM=np.stack((rmm1,rmm2),axis=1)
    #print(RMM.shape)
    #print(RMM[1,:])
    return RMM
#计算RMM指数
def cal_rmm(filename):
    print("date:",filename)
    #根据文件名中的日期打开相对应的三个文件
    path1='/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/OLR/'
    path2='/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U200/'
    path3='/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U850/'
    #读取数据
    data1=netCDF4.Dataset(path1+filename+'_add_120redata_ecmf_olr.nc')
    data2=netCDF4.Dataset(path2+filename+'_add_120redata_ecmf_u200.nc')
    data3=netCDF4.Dataset(path3+filename+'_add_120redata_ecmf_u850.nc')
    data_olr=np.array(data1.variables['olr'][:,:,:])
    data_u200=np.array(data2.variables['u200'][:,:,:])
    data_u850=np.array(data3.variables['u850'][:,:,:])
    #data_olr=np.true_divide(data_olr,3600)
    #一些变量
    path0='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/OLR/'
    data0=netCDF4.Dataset(path0+filename+'_ecmf_olr.nc')
    #时间开始
    time_true=np.array(data0.variables['time'])
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
    #日期数字化
    date_time=int(filename)
    date_yearmonth=date_time//100
    #date_month=date_yearmonth%100
    #如果在闰年并且包含2.29，就使用闰年的气候态
    if (date_yearmonth in run_year)|(date_time in run_year_3):
        #读取气候态
        sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/OLR/'
        sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U200/'
        sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'
        sea_olr=np.load(sea_path1+filename[4:8]+'_run_seasonal_cycle_olr.npy')
        sea_u200=np.load(sea_path2+filename[4:8]+'_run_seasonal_cycle_u200.npy')
        sea_u850=np.load(sea_path3+filename[4:8]+'_run_seasonal_cycle_u850.npy')
        #sea_olr=np.true_divide(sea_olr,3600)
    else:
        #读取气候态
        sea_path1='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/OLR/'
        sea_path2='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U200/'
        sea_path3='/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'
        sea_olr=np.load(sea_path1+filename[4:8]+'_seasonal_cycle_olr.npy')
        sea_u200=np.load(sea_path2+filename[4:8]+'_seasonal_cycle_u200.npy')
        sea_u850=np.load(sea_path3+filename[4:8]+'_seasonal_cycle_u850.npy')
        #sea_olr=np.true_divide(sea_olr,3600)
    
    print(data_olr.shape,data_u200.shape,data_u850.shape)
    print(sea_olr.shape,sea_u200.shape,sea_u850.shape)

    #print(sea_olr[0:4,9,7])
    #print(seasonal_olr[0:4,:,9,7])
    #print(data_olr[0:4,:,9,7])
    #减去气候态
    data_olr=data_olr-sea_olr
    data_u200=data_u200-sea_u200
    data_u850=data_u850-sea_u850
    data_olr=-data_olr
    #print(data_olr[0:4,:,9,7])
    #减去前120天的平均
    #数据长度
    time_length=len(data_olr)
    reduec_120data_olr=avg_120(data_olr,time_length)
    reduec_120data_u200=avg_120(data_u200,time_length)
    reduec_120data_u850=avg_120(data_u850,time_length)
    print(reduec_120data_olr.shape,reduec_120data_u200.shape,reduec_120data_u850.shape)
    #经向平均,变成（180，4，144）
    mean_olr=np.mean(reduec_120data_olr,axis=1)
    mean_u200=np.mean(reduec_120data_u200,axis=1)
    mean_u850=np.mean(reduec_120data_u850,axis=1)
    '''
    test_olr_1=reduec_120data_olr[:,0,:,:]
    mean_test_olr_1=np.mean(test_olr_1,axis=1)
    test_olr_2=reduec_120data_olr[:,1,:,:]
    mean_test_olr_2=np.mean(test_olr_2,axis=1)
    
    print(mean_olr[2,:,9])
    print(mean_test_olr_1[2,9])
    print(mean_test_olr_2[2,9])
    '''
    print(mean_olr.shape,mean_u200.shape,mean_u850.shape)
    #除以标准差 u200=5.3536506 u850=1.9444847 olr=10.035512
    olr_std=16.255560461342792
    u200_std=5.437523725114618
    u850_std=1.9760929603145574
    fina_olr=np.true_divide(mean_olr,olr_std)
    fina_u200=np.true_divide(mean_u200,u200_std)
    fina_u850=np.true_divide(mean_u850,u850_std)
    
    #eof，经验正交分解，使用再分析数据的特征值和特征向量 eof1=46.385155 eof2=45.028522
    lamda_vector=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/pca_lamda_vectors.txt')
    print(lamda_vector.shape)
    RMM=PCs(fina_olr,fina_u850,fina_u200,lamda_vector)
    print(RMM.shape)
    #print(RMM[:,1,:])
    
    #最后判断截取
    if (date_yearmonth in run_year_1)|(date_time in run_year_3):
        true_RMM=RMM[0:46,:]
        print("situation 1")
    elif date_yearmonth in run_year_2:
        true_RMM=RMM[1:47,:]
        print("situation 2")
    else:
        true_RMM=RMM
    #保存
    length=true_RMM.size
    print(length)
    save_path='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_RMM/ecmf/'
    if length==92:
        np.save(save_path+filename+'_ecmf_rmm.npy',true_RMM)
    else:
        print('error!!!!')
        return
for i in range (len(filenames)):
    str_filename=str(int(filenames[i]))
    cal_rmm(filename=str_filename)

print("caluate all success!!!")