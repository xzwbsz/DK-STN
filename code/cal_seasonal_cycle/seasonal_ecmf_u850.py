#导入相关包
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import netCDF4
import datetime
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
#plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
#%matplotlib inline
#定位每个2.29号的time
#输入参数datetime是包含年月日的数字，例如20080311
def find_229_location(date_time):
    #取年份
    date_year=date_time//10000
    start_time=datetime.datetime(1900,1,1,0,0)
    end_time=datetime.datetime(date_year,2,29,0,0)
    date=end_time-start_time
    date_hour=date.days*24
    #print(type(date_hour))
    #print(date_hour)
    return date_hour
#获取文件名
filenames=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/2001_ecmf_date_sort.txt')
print('------')
print(filenames[0])
print(len(filenames))
#封装函数
def cal_seasonal_cycle(filename):
    path='/WdHeDisk/users/zhangnong/MJO/711_test/add_120_re_data/ecmf/U850/'
    #提取文件名的关键字
    date_time=int(filename)
    date_time_fina=date_time+210000
    #print(date_time)
    #print(date_time_fina)
    date_yearmonth=date_time//100
    date_month=date_yearmonth%100
    date_monthday=date_time%10000
    #date_year=date_time//10000
    run_year=np.array([2004,2008,2012,2016,2020])
    print("filename:",filename)
    #1～6月需要考虑2.29情况
    if (1 < date_month < 7)|(date_monthday==121):
        data_16year_1=[]#不包含2.29的版本
        #闰年的几个数组，放在一起，一个有四个，这四个做平均
        data_run_229 = []
        for i in range(date_time,date_time_fina,10000):
            new_filename=str(i)
            date_year=i//10000
            print("date_year:",date_year)
            data=netCDF4.Dataset(path+new_filename+'_add_120redata_ecmf_u850.nc')
            data_olr=np.array(data.variables['u850'])
            data_time=np.array(data.variables['time'])
            print("mean data:",data_olr.shape)
            if date_year in run_year:
                #闰年多了一天，（181，13，144），第一个版本，就是把2.29位置找到，然后删除他，再融合，适合不是闰年的
                #首先找到对应年份的2.29号
                date_hour=find_229_location(i)
                print("date_hour:",date_hour)
                #找到其对应的位置
                location=int(np.argwhere(data_time==int(date_hour)))
                print("location:",location)
                #把2.29的内容压入数组，并在原数组删除
                data_run_229.append(data_olr[location,:,:])
                print(data_olr[location,9,7])
                data_olr=np.delete(data_olr,location,axis=0)
                print("run_year henxin shape:",data_olr.shape)
            data_16year_1.append(data_olr)
        data_16year_1=np.array(data_16year_1)
        data_run_229=np.array(data_run_229)
        print(data_run_229[:,9,7])
        print("first banben:",data_16year_1.shape)
        print("run data shape:",data_run_229.shape)
        #平均16年
        mean_data_1=np.mean(data_16year_1,axis=0)
        mean_data_run=np.mean(data_run_229,axis=0)
        print(mean_data_run[9,7])
        print("mean data1 shape:",mean_data_1.shape)
        print("mean run data shape:",mean_data_run.shape)
        #构造一个适合181天的版本
        mean_data_2=np.insert(mean_data_1,location,mean_data_run,axis=0)
        print("mean data2 shape:",mean_data_2.shape)
        print(mean_data_2[location,9,7])
        #写入文件，前6个月有两个版本，方便后面rmm计算的时候减
        save_filename=str(filename)
        save_filename=save_filename[4:8]
        np.save('/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'+save_filename+'_seasonal_cycle_u850.npy', mean_data_1)
        print("save 1 success!!!")
        np.save('/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'+save_filename+'_run_seasonal_cycle_u850.npy', mean_data_2)
        print("save 2 success!!!")
    else:
        #符合ecmf的7，8，9，10，11，12月情况，不包含二月
        data_16year=[]
        for i in range(date_time,date_time_fina,10000):
            new_filename=str(i)
            data=netCDF4.Dataset(path+new_filename+'_add_120redata_ecmf_u850.nc')
            data_olr=np.array(data.variables['u850'])
            print("mean data:",data_olr.shape)
            data_16year.append(data_olr)
        data_16year=np.array(data_16year)
        print("get 16years data:",data_16year.shape)
        #平均16年
        mean_data=np.mean(data_16year,axis=0)
        print("mean data shape:",mean_data.shape)
        save_filename=str(filename)
        save_filename=save_filename[4:8]
        np.save('/WdHeDisk/users/zhangnong/MJO/711_test/seasonal_cycle/ecmf/U850/'+save_filename+'_seasonal_cycle_u850.npy', mean_data)
        print("save data success!!!")
    return
for i in range (len(filenames)):
    cal_seasonal_cycle(filename=filenames[i])

print("save all success!!!")