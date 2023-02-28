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
print(filenames[len(filenames)-1])
print(len(filenames))
#读取再分析数据的rmm值
re_data_rmm=np.loadtxt('/WdHeDisk/users/zhangnong/MJO/711_test/pca_RMM_1950_2022.txt')
re_data_time=np.array(re_data_rmm[:,0])
print(re_data_time.shape)
#函数，找到模式数据和再分析数据中rmm的对应1
def find_rmm_redata_s2s(filename):
    path='/WdHeDisk/users/zhangnong/MJO/711_test/s2s_data/ecmf/OLR/'
    data=netCDF4.Dataset(path+filename+'_ecmf_olr.nc')
    data_time=int(filename)
    s2s_time=np.array(data.variables['time'])
    len_time=len(s2s_time)
    #定位时间
    location=np.argwhere(re_data_time==data_time)
    print("location:",int(location))
    start_location=int(location)
    fina_location=int(location)+len_time
    #提取时间
    s2s_rmm=np.array(re_data_rmm[start_location:fina_location,1:3])
    print("s2s rmm shape:",s2s_rmm.shape)
    #扩展变成4个
    fina_rmm=s2s_rmm
    print("fina rmm shape:",fina_rmm.shape)
    length=fina_rmm.size
    #保存文件
    save_path='/WdHeDisk/users/zhangnong/MJO/711_test/re_RMM/ecmf/'
    if length==92:
        np.save(save_path+filename+'_ecmf_rmm.npy',fina_rmm)
    else:
        print('error!!!!')
        return
    print("save one!!!")
    return
for i in range (len(filenames)-9):
    str_filename=str(int(filenames[i]))
    find_rmm_redata_s2s(filename=str_filename)

print("caluate all success!!!")
#find_rmm_redata_s2s('20200121')

