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
re_data=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u850_1950-2022.nc')
re_data_olr=np.array(re_data.variables['u850'])
re_data_time=np.array(re_data.variables['time'])
#获取文件名
def get_files():
    for filepath,dirnames,filenames in os.walk(r'/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/U850/'):
        for filename in filenames:
            print (filename)
    return filenames

filenames=get_files()
#读取数据,写个函数
def read_data(filename):
    path='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/U850/'
    new_path='/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U850/'
    data=netCDF4.Dataset(path+filename)
    print("date:",filename)
    #分别提取数据
    lat=np.array(data.variables['latitude'])
    lon=np.array(data.variables['longitude'])
    #插入前120天
    time=np.array(data.variables['time'])
    data_olr=np.array(data.variables['u850'][:,:,:])
    #定位到再分析数据位置
    location=np.argwhere(re_data_time==time[0])
    print("location:",int(location))
    #考虑闰年
    run_year=np.array([200403,200404,200405,200406,200803,200804,200805,200806,201203,201204,201205,201206,201603,201604,201605,201606,
                      202003,202004,202005,202006])
    run_year_2=np.array([200402,200802,201202,201602,202002])
    run_year_3=np.array([20040121,20080121,20120121,20160121,20200121])
    defind_run=int(filename[0:6])
    true_date=int(filename[0:8])
    #如果受闰年影响，我们需要加121天才能对齐
    if defind_run in run_year:
        fina_location=int(location)
        begin_location=fina_location-121
        #输出一下开始时间
        #start_end_time(re_data,begin_location)
        insert_data=re_data_olr[begin_location:fina_location,:,:]
        print("insert data shape:",insert_data.shape)
        print("olr data shape:",data_olr.shape)
        fina_data_ttr=np.concatenate((insert_data,data_olr),axis=0)
        print("fina data shape:",fina_data_ttr.shape)
    elif (defind_run in run_year_2)|(true_date in run_year_3):
        trans_time=time[45]+24
        mou_location=np.argwhere(re_data_time==trans_time)
        mou_location_int=int(mou_location)
        in_data1=re_data_olr[mou_location_int,:,:]
        insert_data1=np.squeeze(in_data1)
        print("in_data1 shape",in_data1.shape)
        print("insert_data1 shape",insert_data1.shape)
        insert_data1=np.expand_dims(insert_data1,0)
        data_olr1=np.concatenate((data_olr,insert_data1),axis=0)
        #添加前120天
        fina_location=int(location)
        begin_location=fina_location-120
        #输出一下开始时间
        #start_end_time(re_data,begin_location)
        insert_data=re_data_olr[begin_location:fina_location,:,:]
        print("insert data shape:",insert_data.shape)
        print("olr data shape:",data_olr1.shape)
        fina_data_ttr=np.concatenate((insert_data,data_olr1),axis=0)
        print("fina data shape:",fina_data_ttr.shape)  
        time=np.append(time,[trans_time],axis=0) 
    else:
        #添加前120天
        fina_location=int(location)
        begin_location=fina_location-120
        #输出一下开始时间
        #start_end_time(re_data,begin_location)
        insert_data=re_data_olr[begin_location:fina_location,:,:]
        print("insert data shape:",insert_data.shape)
        print("olr data shape:",data_olr.shape)
        fina_data_ttr=np.concatenate((insert_data,data_olr),axis=0)
        print("fina data shape:",fina_data_ttr.shape)
    length=len(fina_data_ttr)
    print("length:",length)
    #构造新的时间函数，需要拼接，要不然越界
    first_time=np.array(re_data_time[begin_location:fina_location])
    new_time=np.concatenate((first_time,time),axis=0)
    #构造新的nc文件
    da_olr=netCDF4.Dataset(new_path+filename[0:8]+'_add_120redata_ecmf_u850.nc','w',format='NETCDF4')
    da_olr.createDimension('latitude', 13)  # 创建坐标点
    da_olr.createDimension('longitude', 144)
    da_olr.createDimension('time',length)
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
    da_olr.variables['time'][:]=new_time[:]
    #设置u200变量
    olr =da_olr.createVariable('u850','f',('time','latitude','longitude'))
    olr.units = 'm s**-1'
    #填充数据
    da_olr.variables['u850'][:,:,:]=fina_data_ttr[:,:,:]
    da_olr.close()
    print("success to create olr!!!")
    return 

for i in range (len(filenames)):
    read_data(filename=filenames[i])

print("trans all success!!!")

