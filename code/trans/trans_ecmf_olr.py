#导入相关包
from matplotlib.pyplot import axis
import numpy as np
import os
#import matplotlib.pyplot as plt
import random
import netCDF4
import datetime
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
#plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
#%matplotlib inline
#再分析数据olr
re_data=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/olr_1950-2022.nc')
re_data_olr=np.array(re_data.variables['olr'])
re_data_time=np.array(re_data.variables['time'])
#获取文件名
def get_files():
    for filepath,dirnames,filenames in os.walk(r'/WdHeDisk/users/zhangnong/MJO/use_s2s_data/grib_to_nc/remap_ecmf/OLR/'):
        for filename in filenames:
            print (filename)
    return filenames

filenames=get_files()
#累加，减去前一天的
def delete_add(data):
    #创建一个新数组，和data一样大小
    new_data=np.zeros(data.shape)
    new_data[0,:,:,:]=data[0,:,:,:]
    for i in range(1,len(data)):
        new_data[i,:,:,:]=data[i,:,:,:]-data[i-1,:,:,:]
    #print(new_data[0:10,0,2,7])
    #new_data=np.true_divide(new_data,86400)
    return new_data
#读取数据,写个函数
def read_data(filename):
    path='/WdHeDisk/users/zhangnong/MJO/use_s2s_data/grib_to_nc/remap_ecmf/OLR/'
    new_path='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/OLR/'
    data=netCDF4.Dataset(path+filename)
    #分别提取数据
    lat=np.array(data.variables['lat'])
    lon=np.array(data.variables['lon'])
    #时间要改一下，取59个，然后第一个插入
    time=np.array(data.variables['time'])
    print("time shape:",time.shape)
    #-------------------------------

    #这个要插入再分析数据的第一个
    location=np.argwhere(re_data_time==time[0])
    in_data=re_data_olr[location,:,:]
    in_data=np.squeeze(in_data)
    insert_data=in_data
    print("insert data shape:",insert_data.shape)
    data_ttr=np.array(data.variables['ttr'][1:46,:,:,:])
    #累加，前去前一天的
    new_data_ttr=delete_add(data_ttr)
    new_data_ttr=np.true_divide(new_data_ttr,86400)
    fina_data_ttr=np.insert(new_data_ttr,0,insert_data,axis=0)
    #sst缺失值填为0
    fina_data_ttr[fina_data_ttr==-32767]=0
    sum=np.sum(fina_data_ttr==-32767)
    print("none value:",str(sum))
    print("ttr shape:",fina_data_ttr.shape)
     #样本平均
    fina_data_ttr=np.mean(fina_data_ttr,axis=1)
    #构造新的nc文件u200
    da_olr=netCDF4.Dataset(new_path+filename[8:16]+'_ecmf_olr.nc','w',format='NETCDF4')
    da_olr.createDimension('latitude', 13)  # 创建坐标点
    da_olr.createDimension('longitude', 144)
    da_olr.createDimension('time',46)
    #设置变量
    lat_var = da_olr.createVariable("latitude",'f4',('latitude') )  #添加coordinates  'f'为数据类型，不可或缺
    lat_var.units = 'degrees_north'
    lon_var = da_olr.createVariable("longitude",'f4',('longitude'))  #添加coordinates  'f'为数据类型，不可或缺
    lon_var.units = 'degrees_east'
    time_var=da_olr.createVariable("time",'i4',('time'))
    time_var.units= 'hours since 1900-01-01 00:00:00.0'
    #填充数据
    da_olr.variables['latitude'][:]=lat[:]   #填充数据
    da_olr.variables['longitude'][:]=lon[:]   #填充数据
    da_olr.variables['time'][:]=time[0:46]
    #设置u200变量
    olr =da_olr.createVariable('olr','f',('time','latitude','longitude'))
    olr.units = 'W m**-2'
    #填充数据
    da_olr.variables['olr'][:,:,:]=fina_data_ttr[0:46,:,:]
    da_olr.close()
    print("success to create olr!!!")
    return 

for i in range (len(filenames)):
    read_data(filename=filenames[i])

print("trans all success!!!")

