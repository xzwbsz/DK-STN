#导入相关包
from re import X
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import random
import datetime
import netCDF4
import math
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
from eofs.standard import Eof
from sklearn.decomposition import PCA
#读取数据
data1=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u200_1950-2022.nc')
data2=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u850_1950-2022.nc')
data3=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/olr_1950-2022.nc')
data_u200=np.array(data1.variables['u200'][:,:,:])
data_u850=np.array(data2.variables['u850'][:,:,:])
data_olr=np.array(data3.variables['olr'][:,:,:])
#data_olr=np.true_divide(data_olr,3600)

#读取气候态
seasonal_cycle_u200=np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u200_sample.npy')
seasonal_cycle_u850=np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u850_sample.npy')
seasonal_cycle_olr=np.load('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_olr_sample.npy')
#seasonal_cycle_olr=np.true_divide(seasonal_cycle_olr,3600)
print(seasonal_cycle_u200.shape,seasonal_cycle_u850.shape,seasonal_cycle_olr.shape)

#减去气候态
data_u200=data_u200-seasonal_cycle_u200
data_u850=data_u850-seasonal_cycle_u850
data_olr=data_olr-seasonal_cycle_olr
data_olr=-data_olr
#data_olr=np.true_divide(data_olr,3600)

#返回真实日期
def caluate_time(data_time):
    start_time=datetime.datetime(1900,1,1,0,0)
    fina_time=[]
    for i in range(0,len(data_time)):
        transtime=start_time+datetime.timedelta(hours=int(data_time[i]))
        x=transtime.timetuple().tm_year*10000+transtime.timetuple().tm_mon*100+transtime.timetuple().tm_mday
        #print(x)
        fina_time.append(x)
    final_time=np.array(fina_time)
    return final_time
#减去120天平均值
def avg_120(test_data,time_length):
    new_data=np.zeros(test_data[120:time_length,:,:].shape)
    for i in range(120,time_length):
        mean_120=np.mean(test_data[i-120:i,:,:],axis=0)
        new_data[i-120,:,:]=test_data[i,:,:]-mean_120[:,:]
    final_data=np.array(new_data)
    return final_data
#时间长度
time_data=np.array(data1.variables['time'])
time_length=time_data.size
print(time_length)
#执行减去120天平均的操作
reduec_120data_u200=avg_120(data_u200,time_length)
reduec_120data_u850=avg_120(data_u850,time_length)
reduec_120data_olr=avg_120(data_olr,time_length)
#经向平均
mean_u200=np.mean(reduec_120data_u200,axis=1)
mean_u850=np.mean(reduec_120data_u850,axis=1)
mean_olr=np.mean(reduec_120data_olr,axis=1)

u200_std=np.std(mean_u200)
u850_std=np.std(mean_u850)
olr_std=np.std(mean_olr)
print('std:',u200_std,u850_std,olr_std)

#除以标准差
fina_u200=np.true_divide(mean_u200,u200_std)
fina_u850=np.true_divide(mean_u850,u850_std)
fina_olr=np.true_divide(mean_olr,olr_std)
print(fina_olr.shape,fina_u200.shape,fina_u850.shape)
#三个变量按列拼接，即（时间，144）变成（时间，144*3）
fina_combine=np.concatenate((fina_olr,fina_u850,fina_u200),axis=1)
print(fina_combine.shape)
#print("--------test------------")
#print(fina_combine[25909:25919,10])
#print("--------test------------")

#利用PCA计算
pca=PCA(n_components=2)
pca_rmm=pca.fit_transform(fina_combine)
print("pca_rmm",pca_rmm.shape)
print(pca_rmm[25909:26209,:])
#特征值
lamda=pca.explained_variance_
print("eof1,eof2:",lamda)
print("fangcha:",pca.explained_variance_ratio_)
#特征向量
lamda_vectors=pca.components_
print("-----------vector----------")
print("vector shape:",lamda_vectors.shape)
print(lamda_vectors[:,0:10])
new_lamda_vectors=lamda_vectors.T
print("------------vector------------")
np.savetxt('/WdHeDisk/users/zhangnong/MJO/711_test/pca_lamda_vectors.txt', new_lamda_vectors, fmt="%.8f %.8f")

#最后除以根号下的特征值
eof1=math.sqrt(lamda[0])
eof2=math.sqrt(lamda[1])

print('---------test----------------')
test_rmm=np.matmul(fina_combine,new_lamda_vectors)
test_rmm1=np.true_divide(test_rmm[:,0],eof1)
test_rmm2=np.true_divide(test_rmm[:,1],eof2)
test_RMM=np.stack((test_rmm1,test_rmm2),axis=1)
#np.savetxt('/WdHeDisk/users/zhangnong/MJO/lamda_vector/my_pca_rmm.txt', test_RMM, fmt="%.8f %.8f")
print(test_RMM[25909:26209,:])
print('---------test----------------')

#数组中插入第一列，时间，变换一下
ttime_data=np.array(time_data[120:time_length])
true_data=caluate_time(ttime_data)
#true_data = true_data.reshape(true_data.shape[0], 1)
true_data = true_data.reshape(-1, 1)
print(true_data.shape)
#true_data=true_data.T
print(true_data[0:209])
true_data=true_data.astype(np.float64)
fina_RMM=np.concatenate((true_data,test_RMM),axis=1)
#fina_RMM=np.insert(RMM,0,values=true_data,axis=1)
#fina_RMM=np.c_(true_data,RMM)
print(fina_RMM.shape)
#print(fina_RMM[0:209,0])
np.savetxt('/WdHeDisk/users/zhangnong/MJO/711_test/pca_RMM_1950_2022.txt', fina_RMM, fmt="%d %.8f %.8f")
print("eof1,eof2:",lamda)
print('std:',u200_std,u850_std,olr_std)