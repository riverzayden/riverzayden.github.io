import scipy.stats as ss
import glob
import os
import pandas as pd 
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-one_way_anova", "--one_way_anova", type=bool, default=True,
                help='The number of samples in each batch.')
parser.add_argument("-post_hoc_analysis", "--post_hoc_analysis", type=bool, default=True,
                help='The number of samples in each batch.')
parser.add_argument("-data_path", "--data_path", type=str, default=None,
                help='The number of samples in each batch.')

parser.add_argument("-method", "--method", type=str, default='tukey_hsd',
                help='The number of samples in each batch.')
parser.add_argument("-control_group", "--control_group", type=str, default=None,
                help='The number of samples in each batch.')
parser.add_argument("-split_x", "--split_x", type=bool, default=False,
                help='The number of samples in each batch.')
parser.add_argument("-split_value", "--split_value", type=str, default=None,
                help='The number of samples in each batch.')
parser.add_argument("-calculation_type", "--calculation_type", type=str, default='group_in',
                help='The number of samples in each batch.')
parser.add_argument("-significance_level", "--significance_level", type=float, default=0.95,
                help='The number of samples in each batch.')
args = parser.parse_args()




import scipy.stats as ss
import glob
import os
import pandas as pd 
import numpy as np
from scipy.stats import t
import math
from scipy.stats import norm

def split_data(data_path, split_x, split_value):
    allFiles = glob.glob(os.path.join(data_path, "*.csv"))
    data=[]
    for file_ in allFiles:
        df = pd.read_csv(file_ ,header=None,index_col=None)
        data.append(df)
    data = pd.concat(data)
    data.columns = ['colorby', 'x', 'y']
    if split_x:
        low_indexer = data[data['x']<=split_value].index
        high_indexer = data[data['x']>split_value].index
        data.loc[low_indexer,'split'] = data.loc[low_indexer,'colorby']+'_' +str(0)
        data.loc[high_indexer,'split'] = data.loc[high_indexer,'colorby']+'_' +str(1)
        return data[['y','colorby','split']]
    return data[['y','colorby']]

def qNCDun(p, nu, rho, delta, n = 32, two_sided = True):
    x= GaussLegendre(n)
    nn = 1   
    xx = [p, nu]
    d = qNDBDF(xx[0], rho, xx[1], delta, n=n, x=x)
    return d


def pNDBDF(q, r, nu, delta, xx, n = 32):
    k = len(r)
    x = xx[0]
    w = xx[1]
    y  = 0.5 * x + 0.5
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2)  - (nu/2-1) * np.log(2) + (nu-1) * np.log(y) - nu * y * y / 2
    fx = np.exp(fx)
    y =  y* q
    fndbd_list =[]
    fndbd_list=pNDBD(y,r, delta, n=n, x=xx)
    fy = 0.5*np.array(fndbd_list,dtype='float') * fx
    I  = np.sum(fy * w) 
    b = np.exp(-1)
    a = 0
    x = (b - a) / 2 * x + (b + a) /2 
    y  = -np.log(x) 
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2) - (nu/2-1) * np.log(2) + (nu-1) * np.log(y) - nu * y * y / 2
    fx = np.exp(fx)
    y  = y* q
    fndbd_list =[]
    fndbd_list=pNDBD(y,r, delta, n=n, x=xx)
    fy = (b - a) / 2 * np.array(fndbd_list,dtype='float')  / x * fx
    I =  I + np.sum(fy * w)
    return I


def GaussLegendre(n):
    eig = np.linalg.eig
    i = np.array(range(1,n+1))
    j = np.array(range(1,n))
    mu0 =2 
    b = j/ (4*j**2 - 1)**0.5
    A = np.repeat(0, n*n)
    A = A.astype('float')
    A_index1 = (n+1)*(j-1)+1
    A_index2 = (n + 1) * j -1
    for bindx1, aindx1 in enumerate(A_index1):
        A[aindx1] = b[bindx1]
        A[A_index2[bindx1]] = b[bindx1]
    sd = eig(np.reshape(A,(n,n)))
    w  = sd[1][0]
    w = mu0 * (w**2)
    x = sd[0]
    return x,w


def qNDBDF(p, r, nu, delta,x, n = 32 ):
    k = len(r)
    alpha = 1-p
    alphak = 1-( 1-alpha/2)**(1/k)
    q0 = t.ppf(1 -alphak, nu)+ np.mean(delta) # valor 
    if q0<0 :
        q0 = -q0
    p0 = pNDBDF(np.array([q0]), r, nu, delta, n=n, xx=x)
    print(abs(p0-p))
    if abs(p0-p) >0.04:
        print("여기로 온건가")
        q0 = qNDBD(np.array([p]), r, delta, n=n, x=x)
    maxIt = 5000
    tol = 1e-11
    conv = False
    it = 0
    while conv==False and it <= maxIt:
        p0 = pNDBDF(q0, r, nu, delta, n=n, xx=x)
        q1 = q0 - (p0 - p) / dNDBDF(q0, r, nu, delta, n=n, xx=x)
        if abs(q1-q0) <= abs(q0) * tol:
            conv = True
        q0 = abs(q1)
        it = it + 1     
    return q1





def qNDBD(p, r, delta, x, n = 32):
    k = len(r)
    alpha = 1 - p
    alpha_k = 1 - (1-alpha/2)**(1/k)
    q0 =  norm.ppf(1 - alpha_k) # initial value
    if q0<0:
        q0 = -q0
    maxIt = 5000
    tol = 1e-13
    conv = False
    it = 0
    while conv==False and it <= maxIt:
        q1 = q0 - (pNDBD(q0, r, delta, n=n, x=x) - p) / (dNDBD(q0, r, delta, n=n, x=x))[0][0]
        if (abs(q1-q0) <= tol) :
            conv=True
        q0 = q1
        it = it+1 
    return q1


def pNDBD (q, r, delta,x, n = 32):
    k = len(r)
    y = x[0]/(1-(x[0])**2)
    y= y[:,np.newaxis]
    r = r[np.newaxis,:]
    #print(r**0.5,y)
    yt = np.dot(y,r**0.5)
    if len(q)==1:
        dott = yt.copy()
        jj = q[:,np.newaxis]
        ti = norm.cdf((dott + jj)/((1-r)**0.5))
        ti2 = norm.cdf((dott - jj)/((1-r)**0.5))
        ti = np.array(ti- ti2)
        if np.any(ti<=0):
            ti[ti<=0]= -743.7469 
            ti[ti > 0]= np.log(ti[ti>0])
        else:
            ti =np.log(ti) 
        ti =  np.exp(np.sum(ti,axis=1) +np.squeeze(np.log(norm.pdf(y))))
        I  = ti * ((1+x[0]**2)/(1-x[0]**2)**2)
        I = np.sum(I * x[1])
        return I
    else:
        dott = np.tile(yt.reshape(-1),len(q)).reshape(len(q),yt.shape[0],yt.shape[1])
        jj = q[:,np.newaxis][:,np.newaxis]
        ti = norm.cdf((dott + jj)/((1-r)**0.5))
        ti2 = norm.cdf((dott - jj)/((1-r)**0.5))
        ti = np.array(ti- ti2)
        if np.any(ti<=0):
            ti[ti<=0]= -743.7469 
            ti[ti > 0]= np.log(ti[ti>0])
        else:
            ti =np.log(ti) 
        ti =  np.transpose(np.exp(np.transpose(np.sum(ti,axis=2)) + np.log(norm.pdf(y))))
        I  = ti * ((1+x[0]**2)/(1-x[0]**2)**2)
        I = np.sum(I * x[1],axis=1)
        return I



def dNDBD(q, r, delta, x, n = 32):
    k =len(r)
    y = x[0]/(1-(x[0])**2)
    I_list =[]
    repeat_y = np.repeat(r**0.5,len(y), axis=0).reshape(len(r),len(y)) * y
    repeat_q = np.repeat(repeat_y,len(q),axis=0).reshape(len(q),len(r),len(y))
    cal1 =  (repeat_q +  q[:,np.newaxis][:,np.newaxis])   /((1 - r[0])**0.5)
    cal2 =  (repeat_q -  q[:,np.newaxis][:,np.newaxis])   /((1 - r[0])**0.5)
    normcdf1 = norm.cdf( cal1)
    normcdf2 = norm.cdf( cal2)
    Phy = normcdf1 -normcdf2
    if np.any(Phy<=0):
        Phy[Phy<=0]= -743.7469 
        Phy[Phy > 0]= np.log(Phy[Phy>0])
    else:
        Phy =np.log(Phy) 
    normpdf1  = norm.pdf( cal1)
    normpdf2  = norm.pdf( cal2)
    phy = normpdf1 +normpdf2 
    if np.any(phy<=0):
        phy[phy<=0]= -743.7469 
        phy[phy > 0]= np.log(phy[phy>0])
    else:
        phy =np.log(phy) 
    phy =  phy - 0.5 * np.log(1 - r[0])
    sPhy = np.sum(Phy,axis=1) # sum of Phi's differences  ==> 3 * 3 ( len(r)* len(y))
    sTi = np.sum(np.exp(phy + sPhy[:,np.newaxis] - Phy),axis=1) *norm.pdf(np.transpose(y[:,np.newaxis]))
    I = np.dot(sTi,(((1+x[0]**2)/(1-x[0]**2)**2)*x[1])[:,np.newaxis]) 
    return I


###3222

def pNDBDF(q, r, nu, delta, xx, n = 32):
    k = len(r)
    x = xx[0]
    w = xx[1]
    y  = 0.5 * x + 0.5
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2)  - (nu/2-1) * np.log(2) + (nu-1) * np.log(y) - nu * y * y / 2
    fx = np.exp(fx)
    y =  y* q
    fndbd_list =[]
    fndbd_list=pNDBD2(y,r, delta, n=n, x=xx)
    fy = 0.5*np.array(fndbd_list,dtype='float') * fx
    I  = np.sum(fy * w) 
    b = np.exp(-1)
    a = 0
    x = (b - a) / 2 * x + (b + a) /2 
    y  = -np.log(x) 
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2) - (nu/2-1) * np.log(2) + (nu-1) * np.log(y) - nu * y * y / 2
    fx = np.exp(fx)
    y  = y* q
    fndbd_list =[]
    fndbd_list=pNDBD2(y,r, delta, n=n, x=xx)
    fy = (b - a) / 2 * np.array(fndbd_list,dtype='float')  / x * fx
    I =  I + np.sum(fy * w)
    return I

def pNDBD2(q, r, delta,x, n = 32):
    # q가 리스트! 
    k = len(r)
    y = x[0]/(1-(x[0])**2)
    yt = np.dot(y[:,np.newaxis],r[np.newaxis,:]**0.5)
    dott = np.tile(yt.reshape(-1),len(q)).reshape(len(q),len(q),len(r))
    jj = q[:,np.newaxis][:,np.newaxis]
    ti = norm.cdf(((yt+jj)/((1-r)**0.5)))
    ti = ti-  norm.cdf(((yt-jj)/((1-r)**0.5)))
    
    if np.any(ti<=0):
        ti[ti<=0]= -743.7469 
        ti[ti > 0]= np.log(ti[ti>0])
    else:
        ti =np.log(ti) 
    ti =np.exp(np.sum(ti,axis=2) + np.log(norm.pdf(y)))
    I  = ti * ((1+x[0]**2)/(1-x[0]**2)**2)
    I = np.sum(I * x[1],axis=1)
    return I


def dNDBDF(q, r, nu, delta, xx, n = 32):
    # r= > df1 ==> 그룹개수
    k = len(r)
    x = xx[0]
    w = xx[1]
    # computing integral in [0,1]
    y  = 0.5 * x + 0.5 # from [-1,1] to [0, 1]
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2) - (nu/2-1) * np.log(2) +  \
           (nu-1) * np.log(y) - nu * y * y / 2 + np.log(y) # + jacobian
    fx = np.exp(fx)
    y  = y * q
    fy_list =[]
    fy = 0.5* dNDBD(y, r, delta, n=n, x=xx) 
    I = np.sum(np.squeeze(fy) * fx *w)
    b = np.exp(-1)
    a = 0
    x = (b - a) / 2 * x + (b + a) / 2 # from [-1,1] to [0, exp(-1)]    
    y  = -np.log(x) # from [0, exp(-1)] to [1, +infty)
    fx = nu/2 * np.log(nu) - math.lgamma(nu/2) - (nu/2-1) * np.log(2) +  \
         (nu-1) * np.log(y) - nu * y * y / 2 + np.log(y) # + jacobian
    fx = np.exp(fx)
    y  = y * q
    fy = dNDBD(y, r, delta, n=n, x=xx) 
    fy = (b - a) / 2 * np.squeeze(fy) / x * fx
    I  = I + np.sum(fy * w)
    return I



def pNCDun(q, nu, rho, delta, n = 32, two_sided = True):
    x= GaussLegendre(n)
    nn = 1
    xx = np.array([q, nu])
    rr = pNDBDF(xx[0], rho, xx[1], delta, n=n, xx=x)
    return rr

        
        
def post_hoc_test(type, data, alpha=0.95, post_hoc_analysis=True, control_group=None, one_way_anova=False ):
    import itertools as it
    alpha= 1-alpha
    group_unique = set(data['colorby'])
    group_size = len(group_unique)
    if len(group_unique)==1:
        group_size = len(group_unique)
        combs = it.combinations(range(group_size), 2)
        data = pd.DataFrame({'group1':list(group_unique), 'group2':[np.nan]})
        if one_way_anova:
            data['f_value'] = np.nan
            data['p_value'] = np.nan
        if post_hoc_analysis:
            data['meandiff'] = np.nan
            data['crit_value'] = np.nan
            data['lower'] = np.nan
            data['upper'] = np.nan
            data['post_p_value'] = np.nan
        return data
    if type=='tukey_hsd':
        '''alpha : {0.05, 0.01}'''
        from statsmodels.stats.libqsturng import qsturng
        import rpy2.robjects as robjects
        ptukey = robjects.r['ptukey']
        #data = pd.DataFrame({'group':groups,'value':values})
        total_n = len(data)
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, group_size-1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])
            tukey_value = qsturng(1-alpha,group_size,df2  )
            data['mean_1'] = value_mean_i.values[data['group1']]
            data['mean_2'] = value_mean_i.values[data['group2']]
            data['n_1'] = n_i.values[data['group1']]
            data['n_2'] = n_i.values[data['group2']]
            data['meandiff'] = data['mean_1'] - data['mean_2']
            data['std_error'] = np.sqrt((mse/2) * ( 1/data['n_1']+ 1/data['n_2']))
            data['crit_value'] = data['std_error']*tukey_value
            def lower_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['lower'] = -high
                else:
                    x['lower'] = low
                return x['lower']
            def upper_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['upper'] = -low
                else:
                    x['upper'] = high
                return x['upper']
            data['lower'] = data.apply(lower_fun,axis=1)
            data['upper'] = data.apply(upper_fun,axis=1)
            #data['p_value'] = psturng(abs(data['meandiff']) / data['std_error'],group_size,df2)
            def p_value_fun(x):
                return round(1-ptukey(abs(x['meandiff']) / x['std_error'],group_size,df2)[0],4)
            data['post_p_value'] = data.apply(p_value_fun,axis=1)
#             def reject_fun(x):
#                 if x['post_p_value']<alpha:
#                     return True
#                 else:
#                     return False 
#             data['reject'] = data.apply(reject_fun, axis=1)

            combs = list(it.combinations(range(group_size), 2))
            if one_way_anova:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value'],
                                      'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                       'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                                       'crit_value': data['crit_value'],
                                      'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
            else:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 
                                      'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                       'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                                       'crit_value': data['crit_value'],
                                      'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]                
            return data
        else:
            combs = list(it.combinations(range(group_size), 2))
            data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
            data = pd.concat([data,data_2],sort=False)
            data = data.sort_values(['group1','group2'])
            unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
            data['group1'] = data['group1'].map(unique_dict)
            data['group2'] = data['group2'].map(unique_dict)
            return data

        
    elif type=='scheffe':
        import itertools as it
        total_n = len(data )
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        value_var_i = data.groupby('colorby').var()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, group_size-1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)       
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])
            data['mean_1'] = value_mean_i.values[data['group1']]
            data['mean_2'] = value_mean_i.values[data['group2']]
            data['n_1'] = n_i.values[data['group1']]
            data['n_2'] = n_i.values[data['group2']]
            data['meandiff'] = data['mean_1'] - data['mean_2']
            data['std_error'] = np.sqrt((mse) * ( 1/data['n_1']+ 1/data['n_2']))
            data['crit_value'] = np.sqrt((group_size- 1.) * ss.f.ppf(q=1-alpha, dfn=group_size- 1, dfd=df2)) \
                                * data['std_error'] 
            def lower_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['lower'] = -high
                else:
                    x['lower'] = low
                return x['lower']
            def upper_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['upper'] = -low
                else:
                    x['upper'] = high
                return x['upper']
            data['lower'] = data.apply(lower_fun,axis=1)
            data['upper'] = data.apply(upper_fun,axis=1)

            def p_value_fun(x):
                return round(ss.f.sf((x['meandiff'])**2 / ((x['std_error'])**2*(group_size-1)),group_size - 1,df2),4)
            #p_values = ss.f.sf(f_value, group_size - 1., n - group_size)
            data['post_p_value'] = data.apply(p_value_fun,axis=1)
            # def reject_fun(x):
            #     if x['p_value']<alpha:
            #         return True
            #     else:
            #         return False
            # data['reject'] = data.apply(reject_fun, axis=1)
            combs = list(it.combinations(range(group_size), 2))
            if one_way_anova:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],
                                    'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                    'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                                    'crit_value': data['crit_value'],
                                    'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
            else:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 
                                      'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                       'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                                       'crit_value': data['crit_value'],
                                      'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]   
            return data  
        else:
            combs = list(it.combinations(range(group_size), 2))
            data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
            data = pd.concat([data,data_2],sort=False)
            data = data.sort_values(['group1','group2'])
            unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
            data['group1'] = data['group1'].map(unique_dict)
            data['group2'] = data['group2'].map(unique_dict)
            return data
        
    elif type=='duncan':
        '''R Package DescTools:PostHocTest 
           R Pacakge Reference : library(agricolae) : duncan.test   '''
        import rpy2.robjects as robjects
        from statsmodels.stats.libqsturng import qsturng
        ptukey = robjects.r['ptukey']
        total_n = len(data)
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        value_var_i = data.groupby('colorby').var()
        mean_rank_i = value_mean_i.rank()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, group_size-1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])     
                data['mean_1'] = value_mean_i.values[data['group1']]
                data['mean_2'] = value_mean_i.values[data['group2']]
                data['n_1'] = n_i.values[data['group1']]
                data['n_2'] = n_i.values[data['group2']]
                data['meandiff'] = data['mean_1'] - data['mean_2']
                data['std_error'] = np.sqrt((mse/2) * ( 1/data['n_1']+ 1/data['n_2']))
                data['mean_rank_1'] = mean_rank_i.values[data['group1']]
                data['mean_rank_2'] = mean_rank_i.values[data['group2']]
                # abs(rank1- rank2) +1
                data['minus_rank_plus1']  = abs(data['mean_rank_1']  - data['mean_rank_2']) +1
                # crit_value = stderror + qsturng ( (1-alpha)**(minus_rank) , minus_rank_plus1, df2 )
                data['crit_value'] = data['std_error'] * qsturng((1-alpha)**(data['minus_rank_plus1']-1), data['minus_rank_plus1'],df2) 
                def lower_fun(x):
                    high = abs(x['meandiff'])  + x['crit_value']
                    low = abs(x['meandiff'])  - x['crit_value']
                    if x['meandiff']<0:
                        x['lower'] = -high
                    else:
                        x['lower'] = low
                    return x['lower']
                def upper_fun(x):
                    high = abs(x['meandiff'])  + x['crit_value']
                    low = abs(x['meandiff'])  - x['crit_value']
                    if x['meandiff']<0:
                        x['upper'] = -low
                    else:
                        x['upper'] = high
                    return x['upper']
                data['lower'] = data.apply(lower_fun,axis=1)
                data['upper'] = data.apply(upper_fun,axis=1)
                # pvalue = abs(mean_diff) / std_error 
                def p_value_fun(x):
                    pvalue = ptukey(abs(x['meandiff']) / x['std_error'],x['minus_rank_plus1'],df2)[0]
                    pvalue = 1-(pvalue)**(1/(x['minus_rank_plus1']-1))
                    return round(pvalue,4)
                data['post_p_value'] = data.apply(p_value_fun,axis=1)
                # def reject_fun(x):
                #     if x['p_value']<alpha:
                #         return True
                #     else:
                #         return False
                # data['reject'] = data.apply(reject_fun, axis=1)
        
                combs = list(it.combinations(range(group_size), 2))
                if one_way_anova:
                    data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],  'f_value':data['f_value'],'p_value':data['p_value'],
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 
                               'mean_rank_1':data['mean_rank_2'],'mean_rank_2':data['mean_rank_1'],
                               'minus_rank_plus1':data['minus_rank_plus1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                    data = pd.concat([data,data_2],sort=False)
                    data = data.sort_values(['group1','group2'])
                    unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                    data['group1'] = data['group1'].map(unique_dict)
                    data['group2'] = data['group2'].map(unique_dict)
                    data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
                else:
                    data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],
                                        'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                        'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 
                                        'mean_rank_1':data['mean_rank_2'],'mean_rank_2':data['mean_rank_1'],
                                        'minus_rank_plus1':data['minus_rank_plus1'], 'post_p_value':data['post_p_value'],
                                        'crit_value': data['crit_value'],
                                        'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                    data = pd.concat([data,data_2],sort=False)
                    data = data.sort_values(['group1','group2'])
                    unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                    data['group1'] = data['group1'].map(unique_dict)
                    data['group2'] = data['group2'].map(unique_dict)
                    data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]    
                return data
            else:
                combs = list(it.combinations(range(group_size), 2))
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                return data
            
            
    elif type=='lsd':
        '''fishers lsd'''
        total_n = len(data)
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        value_var_i = data.groupby('colorby').var()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, group_size-1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])
            lsd_value = ss.t.ppf(1-alpha/2,df2)
            data['mean_1'] = value_mean_i.values[data['group1']]
            data['mean_2'] = value_mean_i.values[data['group2']]
            data['n_1'] = n_i.values[data['group1']]
            data['n_2'] = n_i.values[data['group2']]
            data['meandiff'] = data['mean_1'] - data['mean_2']
            data['std_error'] = np.sqrt(mse * ( 1/data['n_1']+ 1/data['n_2']))
            data['crit_value'] = data['std_error']*lsd_value
            def lower_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['lower'] = -high
                else:
                    x['lower'] = low
                return x['lower']
            def upper_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['upper'] = -low
                else:
                    x['upper'] = high
                return x['upper']
            data['lower'] = data.apply(lower_fun,axis=1)
            data['upper'] = data.apply(upper_fun,axis=1)
            # tvalue => abs(meandiff) / stdiff
            data['post_p_value'] = ss.t.sf(abs(data['meandiff']) / data['std_error'],df2)*2
            # def reject_fun(x):
            #     if x['p_value']<alpha:
            #         return True
            #     else:
            #         return False
            # data['reject'] = data.apply(reject_fun, axis=1)
        
            combs = list(it.combinations(range(group_size), 2))
            if one_way_anova:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],  'f_value':data['f_value'],'p_value':data['p_value'],
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
            else:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]       
            return data
        else:
            combs = list(it.combinations(range(group_size), 2))
            data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
            data = pd.concat([data,data_2],sort=False)
            data = data.sort_values(['group1','group2'])
            unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
            data['group1'] = data['group1'].map(unique_dict)
            data['group2'] = data['group2'].map(unique_dict)
            return data

    elif type=='bonferroni':
        '''bonferroni correction test'''
        total_n = len(data )
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        value_var_i = data.groupby('colorby').var()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, group_size-1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])
            K = group_size*(group_size-1)
            # 1-alpha/(r*(r-1))
            bonf_value = ss.t.ppf(1-alpha/K,df2)
            data['mean_1'] = value_mean_i.values[data['group1']]
            data['mean_2'] = value_mean_i.values[data['group2']]
            data['n_1'] = n_i.values[data['group1']]
            data['n_2'] = n_i.values[data['group2']]
            data['meandiff'] = data['mean_1'] - data['mean_2']
            data['std_error'] = np.sqrt(mse * ( 1/data['n_1']+ 1/data['n_2']))
            data['crit_value'] = data['std_error']*bonf_value
            def lower_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['lower'] = -high
                else:
                    x['lower'] = low
                return x['lower']
            def upper_fun(x):
                high = abs(x['meandiff'])  + x['crit_value']
                low = abs(x['meandiff'])  - x['crit_value']
                if x['meandiff']<0:
                    x['upper'] = -low
                else:
                    x['upper'] = high
                return x['upper']
            data['lower'] = data.apply(lower_fun,axis=1)
            data['upper'] = data.apply(upper_fun,axis=1)
            # tvalue => abs(meandiff) / stdiff
            data['post_p_value'] = ss.t.sf(abs(data['meandiff']) / data['std_error'],df2)*K
            def p_value_fun(x):
                if x>1:
                    x=1
                return round(x,4)
            data['post_p_value'] = data['post_p_value'].apply(p_value_fun)
            # def reject_fun(x):
            #     if x['p_value']<alpha:
            #         return True
            #     else:
            #         return False
            # data['reject'] = data.apply(reject_fun, axis=1)
            
            combs = list(it.combinations(range(group_size), 2))
            if one_way_anova:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],'f_value':data['f_value'],'p_value':data['p_value'],
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
            else:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0],
                                    'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                                    'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                                    'crit_value': data['crit_value'],
                                    'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]                
            return data
        else:
            combs = list(it.combinations(range(group_size), 2))
            data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
            data = pd.concat([data,data_2],sort=False)
            data = data.sort_values(['group1','group2'])
            unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
            data['group1'] = data['group1'].map(unique_dict)
            data['group2'] = data['group2'].map(unique_dict)
            return data
        
    # 그냥 다구했음. 원래는 대조그룹 통하여 구하는 것   ( group1을 기준으로 가져가면됨.)
    elif type=='dunnett':
        total_n = len(data )
        n_i = data.groupby('colorby').count() 
        n = n_i.sum()
        value_mean_i = data.groupby('colorby').mean() 
        value_var_i = data.groupby('colorby').var()
        value_var_i = data.groupby('colorby').var()
        mse = (1. / (n - group_size) * np.sum(value_var_i * (n_i - 1.))).values[0]
        print(mse)
        df2 = total_n - group_size
        df1 = group_size-1
        combs = it.combinations(range(group_size), 2)
        if one_way_anova:
            mstr = np.sum(n_i * (value_mean_i - data['y'].mean())**2) / (group_size-1)
            f_value = mstr/mse
            p_value =  (1-ss.f.cdf(f_value, df1, df2))[0]
            data = pd.DataFrame(combs,columns=['group1','group2'])
            data['f_value']= round(f_value.values[0],4)
            data['p_value'] = round(p_value,4)
        if post_hoc_analysis:
            if not one_way_anova:
                data = pd.DataFrame(combs,columns=['group1','group2'])
            data['mean_1'] = value_mean_i.values[data['group1']]
            data['mean_2'] = value_mean_i.values[data['group2']]
            data['n_1'] = n_i.values[data['group1']]
            data['n_2'] = n_i.values[data['group2']]
            data['meandiff'] = data['mean_2'] - data['mean_1']
            data['std_error'] = np.sqrt(mse * ( 1/data['n_1']+ 1/data['n_2']))
            qfrompdunnett =  qNCDun(p=1-alpha, nu=total_n-group_size, 
                                    rho=np.repeat(0.5,group_size-1), delta=np.repeat(0,group_size-1),n=32, two_sided=True)
            def lower_fun(x):
                high = abs(x['meandiff'])  + qfrompdunnett * x['std_error']
                low = abs(x['meandiff'])  - qfrompdunnett * x['std_error']
                if x['meandiff']<0:
                    x['lower'] = -high
                else:
                    x['lower'] = low
                return x['lower']
            def upper_fun(x):
                high = abs(x['meandiff'])  + qfrompdunnett * x['std_error']
                low = abs(x['meandiff'])  - qfrompdunnett * x['std_error']
                if x['meandiff']<0:
                    x['upper'] = -low
                else:
                    x['upper'] = high
                return x['upper']
            def p_value_fun(x):
                p = abs(x['meandiff']) / x['std_error']
                pval = 1-pNCDun(q=p, nu=df2, rho=np.repeat(0.5,df1), delta=np.repeat(0,df1), n=32,two_sided=True)
                return round(pval,4)
            data['post_p_value'] = data.apply(p_value_fun, axis=1)
            # def reject_fun(x):
            #     if x['p_value']<alpha:
            #         return True
            #     else:
            #         return False
            # data['reject'] = data.apply(reject_fun, axis=1)
            data['crit_value'] = qfrompdunnett* data['std_error']
            data['lower'] = data.apply(lower_fun,axis=1)
            data['upper'] = data.apply(upper_fun,axis=1)
            combs = list(it.combinations(range(group_size), 2))
            if one_way_anova:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value'],
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2', 'f_value','p_value','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]
            else:
                data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 
                              'n_1':data['n_2'],'n_2':data['n_1'], 'meandiff':-data['meandiff'],
                               'mean_1':data['mean_2'], 'mean_2':data['mean_1'], 'post_p_value':data['post_p_value'],
                               'crit_value': data['crit_value'],
                              'std_error':data['std_error'], 'lower':-data['upper'], 'upper':-data['lower']})
                data = pd.concat([data,data_2],sort=False)
                data = data.sort_values(['group1','group2'])
                unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
                data['group1'] = data['group1'].map(unique_dict)
                data['group2'] = data['group2'].map(unique_dict)
                data = data[['group1', 'group2','meandiff', 'crit_value', 'lower', 'upper', 'post_p_value']]                
            return data
        else:
            combs = list(it.combinations(range(group_size), 2))
            data_2 = pd.DataFrame({'group1':np.array(combs)[:,1], 'group2':np.array(combs)[:,0], 'f_value':data['f_value'],'p_value':data['p_value']})
            data = pd.concat([data,data_2],sort=False)
            data = data.sort_values(['group1','group2'])
            unique_dict = {inx:i for inx, i in enumerate(n_i.index)}
            data['group1'] = data['group1'].map(unique_dict)
            data['group2'] = data['group2'].map(unique_dict)
            return data
        
def exe(data_path, split_value, one_way_anova, post_hoc_analysis,significance_level, method,  split_x, calculation_type, control_group=None):
    if one_way_anova==False and post_hoc_analysis==False:
        return 1
    else:
        if split_x :
            print("쪼개기 있음.")
            data = split_data(data_path=data_path,split_value=split_value, split_x=split_x )
            if calculation_type =='group_in':
                print("그룹안에서 하나씩 하고 포문돌리기 ( 단 그룹안에 개수가 0인거 파악해야한다.)")
                unique_name = set(data['colorby'])
                data_list =[]
                for i in unique_name:
                    part_data = data[data['colorby']==i]
                    part_data = part_data[['y','split']]
                    part_data  = part_data.rename(columns={'split':'colorby'})
                    print(set(part_data['colorby']))
                    r = post_hoc_test(type='tukey_hsd', data = part_data,  post_hoc_analysis=post_hoc_analysis, one_way_anova=one_way_anova  )  
                    data_list.append(r)
                    data = data.drop(part_data.index)
                output = pd.concat(data_list)
                return output
            else:
                print("그룹전체로 한방에 ")
                data = data[['y','split']]
                data  = data.rename(columns={'split':'colorby'})
                r = post_hoc_test(type='tukey_hsd', data = data,  post_hoc_analysis=post_hoc_analysis, one_way_anova=one_way_anova  )  
                return r
        else:
            print("쪼개기 없음, 하던데로 하면됨 ")
            data = split_data(data_path=data_path,split_value=split_value, split_x=split_x )
            r = post_hoc_test(type=method, data = data,  post_hoc_analysis=post_hoc_analysis, one_way_anova=one_way_anova  )  
            return r
        
        

        
        
if __name__ == "__main__":        
    one_way_anova = args.one_way_anova
    post_hoc_analysis = args.post_hoc_analysis
    data_path = args.data_path
    method = args.method
    control_group = args.control_group
    split_x = args.split_x
    split_value = args.split_value
    calculation_type = args.calculation_type
    significance_level = args.significance_level
    print(one_way_anova ,post_hoc_analysis, data_path,method,  control_group, split_x , split_value,  calculation_type,significance_level )
    output = exe(data_path=data_path, split_value=split_value, one_way_anova=one_way_anova,
                 post_hoc_analysis=post_hoc_analysis, significance_level=significance_level, method=method,
                 split_x=split_x, calculation_type=calculation_type, control_group=control_group)