#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 2024

@author: MegNParker

https://www.statsmodels.org/stable/gettingstarted.html

"""
#%%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels as statsmodels
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels as statsmodels
from statsmodels.formula.api import ols


# Import date class from datetime module
from datetime import date
today = date.today()

#%% IMPORT DATAFILES
#Navigate to the folder where data are located. 
dir_base="..." #directory that contains test meal csv file. 

df_testmeal_raw = pd.read_csv(f"{dir_base}/energy_intake_data.csv") 
df_testmeal_raw.columns
df_testmeal_raw.shape
                                                         
#check number of included sub
subj_count=df_testmeal_raw.pivot_table(index=['subjid'],aggfunc='size')
n_sub=subj_count.shape[0] 

#%% Appy necessary transformations to regressors. 

df_testmeal_raw ['LOC_YN'] = pd.to_numeric(df_testmeal_raw ['LOC_YN'], errors='coerce')

# make sure vars are numeric 
df_testmeal_raw ['TOT_FAT'] = pd.to_numeric(df_testmeal_raw ['TOT_FAT'], errors='coerce')
df_testmeal_raw['TOT_LEAN'] = pd.to_numeric(df_testmeal_raw['TOT_LEAN'], errors='coerce')
df_testmeal_raw['TOT_MASS'] = pd.to_numeric(df_testmeal_raw ['TOT_MASS'], errors='coerce')
df_testmeal_raw ['TOTAL_CAL_CONSUME'] = pd.to_numeric(df_testmeal_raw['TOTAL_CAL_CONSUME'], errors='coerce')
df_testmeal_raw['TOTAL_PRO_PCT_CONSUME'] = pd.to_numeric(df_testmeal_raw['TOTAL_PRO_PCT_CONSUME'], errors='coerce')
df_testmeal_raw ['TOTAL_FAT_PCT_CONSUME'] = pd.to_numeric(df_testmeal_raw ['TOTAL_FAT_PCT_CONSUME'], errors='coerce')
df_testmeal_raw ['TOTAL_CARB_PCT_CONSUME'] = pd.to_numeric(df_testmeal_raw ['TOTAL_CARB_PCT_CONSUME'], errors='coerce')

#make sure that using "coerce" didn't force any NaN values.  
df_testmeal_raw.isnull().any()

#  converting g to kg for DXA for tot fat and tot lean vars:
df_testmeal_raw['TOT_FAT'] = df_testmeal_raw['TOT_FAT'] / 1000    
df_testmeal_raw['TOT_LEAN'] = df_testmeal_raw['TOT_LEAN'] / 1000  

# computing fat % based on tot fat mass and tot mass
df_testmeal_raw['FAT_PERCENT'] = df_testmeal_raw['TOT_FAT']/df_testmeal_raw['TOT_MASS']

#transforming the new dexa fat % var for normality as it is a percentage
df_testmeal_raw['FAT_PERCENT_ARCSIN'] = np.arcsin(np.sqrt(df_testmeal_raw['FAT_PERCENT']))

#arc sign sqrt the newly divided test meal %s
df_testmeal_raw['TOTAL_PRO_PCT_CONSUME_ARCSIN'] = np.arcsin(np.sqrt((df_testmeal_raw['TOTAL_PRO_PCT_CONSUME']/100)))
df_testmeal_raw['TOTAL_FAT_PCT_CONSUME_ARCSIN'] = np.arcsin(np.sqrt((df_testmeal_raw['TOTAL_FAT_PCT_CONSUME']/100)))
df_testmeal_raw['TOTAL_CARB_PCT_CONSUME_ARCSIN'] = np.arcsin(np.sqrt((df_testmeal_raw['TOTAL_CARB_PCT_CONSUME']/100)))

# make sure that transformations didn't force any NaN values.
df_testmeal_raw.isnull().any()
df_testmeal_raw=df_testmeal_raw.drop(columns=['TOTAL_PRO_PCT_CONSUME' ,'TOTAL_FAT_PCT_CONSUME','TOTAL_CARB_PCT_CONSUME','TOT_FAT','TOT_MASS','FAT_PERCENT'])

#%% dummy code race and ethnicity  

df_testmeal_raw['Nonwhite'] = 0

df_testmeal_raw.loc[(df_testmeal_raw['RACE'] == 'WHITE') & (df_testmeal_raw['ETHNICITY'] != 'LATINO OR HISPANIC'), 'Nonwhite'] = 1

#check number of included sub
df_testmeal_raw ['Nonwhite'] = pd.to_numeric(df_testmeal_raw['Nonwhite'], errors='coerce')

#check that the number of non Hispanic White and other look correct
subj_count=df_testmeal_raw.pivot_table(df_testmeal_raw,index=['Nonwhite'],aggfunc='size')

# Drop the original 'race' and 'ethnicity' columns 
df_testmeal_raw = df_testmeal_raw.drop(['RACE', 'ETHNICITY'], axis=1)

#%%
#check skewness and kurtosis. 
pd_describe=df_testmeal_raw.describe()
print(pd_describe)
df_testmeal_raw.skew(axis=0,numeric_only=True)
df_testmeal_raw.kurtosis(axis=0,numeric_only=True)

#%% convert dataset from long format to wide format 
df_testmeal_raw_wide=df_testmeal_raw.pivot(index='subjid',columns='INTERVAL_NAME',values=['TOTAL_CAL_CONSUME', 'TOTAL_PRO_PCT_CONSUME_ARCSIN','TOTAL_FAT_PCT_CONSUME_ARCSIN', 'TOTAL_CARB_PCT_CONSUME_ARCSIN'])
df_testmeal_raw_wide=df_testmeal_raw.pivot(index='subjid',columns='INTERVAL_NAME')

# Flatten MultiIndex columns - this allows for interval_name (pre/post) to become part of each column name treating them as the same variable and not separate vars
df_testmeal_raw_wide.columns = ['_'.join(col).rstrip('_') for col in df_testmeal_raw_wide.columns.values]

# Reorder columns so that 'pre' vars come before 'post' vars in new dataset. ALSO, only include covars as "pre" and outcome vars as pre and post:
df_testmeal_raw_wide = df_testmeal_raw_wide[['tx_condition_pre', 'Nonwhite_pre', 'age_pre', 'LOC_YN_pre',  
                   'height_pre', 'TOT_LEAN_pre',
                   'FAT_PERCENT_ARCSIN_pre',
                   'TOTAL_CAL_CONSUME_pre', 'TOTAL_CAL_CONSUME_post', 
                   'TOTAL_PRO_PCT_CONSUME_ARCSIN_pre', 'TOTAL_PRO_PCT_CONSUME_ARCSIN_post',
                   'TOTAL_FAT_PCT_CONSUME_ARCSIN_pre', 'TOTAL_FAT_PCT_CONSUME_ARCSIN_post',
                   'TOTAL_CARB_PCT_CONSUME_ARCSIN_pre',
                   'TOTAL_CARB_PCT_CONSUME_ARCSIN_post'
                   ]]

print(df_testmeal_raw_wide)
df_testmeal_raw_wide.columns #check- did it reorder correctly/do we have all the columns we need?


#%% RUN General linear MODEL TESTING THE EFFECT OF CONDITION ON CHANGE IN test meal intake

dv="TOTAL_CAL_CONSUME" #to initialize function. 

def run_ols (dv):
    model_total_cal = f"{dv}_post ~ {dv}_pre+ C(tx_condition_pre)+C(LOC_YN_pre)+ C(Nonwhite_pre) +TOT_LEAN_pre +FAT_PERCENT_ARCSIN_pre +age_pre +height_pre"
    total_cal = smf.ols(model_total_cal,data=df_testmeal_raw_wide).fit()
    total_cal.summary() 
    
    #save paramater estimates.        
    param=pd.DataFrame(total_cal.params)
    param=param.rename(columns={0: "beta"})
    bse=pd.DataFrame(total_cal.bse)
    bse=bse.rename(columns={0: "se"})
    ci=pd.DataFrame(total_cal.conf_int())
    ci=ci.rename(columns={0: "95_lower",1:'95_upper'})
    mixed_summary = pd.concat([param, bse,ci],axis=1)
    mixed_summary['p']=total_cal.pvalues
    
    #compute SD for each tx group based on residuals from the total_cal. 
    resid=pd.DataFrame(total_cal.resid)
    resid=resid.rename(columns={0:'resid'})
    resid['resid_sq']=resid['resid']**2
    temp=pd.concat([resid,df_testmeal_raw_wide], axis =1)
    temp1=temp.pivot_table(index=['tx_condition_pre'],values=['resid','resid_sq'],aggfunc='sum')
    temp2=temp.pivot_table(index=['tx_condition_pre'],values=['resid','resid_sq'],aggfunc='size')
    temp2=temp2.rename_axis('subj_count')
    temp1=pd.concat([temp1,temp2], axis =1)
    temp1=temp1.rename(columns={0:'subj_count'})
    n_ctrl=temp1['subj_count'].loc[0]#control
    n_active=temp1['subj_count'].loc[1]#active
    temp1['var']=temp1['resid_sq']/(temp1['subj_count']-1)
    temp1['SD']=np.sqrt(temp1['var'])
    temp1['SD_sq']=(temp1['SD']**2)
    SD_ctrl=temp1['SD_sq'].loc[0]#control
    SD_active=temp1['SD_sq'].loc[1]#active   
    SD_pooled= math.sqrt((((n_ctrl-1)*SD_ctrl)+((n_active-1)*SD_active))/(n_ctrl+n_active-2))
    
    #compute predicted values and obtain EMM
    #list of possible comparison, condition and LOC combos.
    tx_condition=[1,0,1,0,1,0,1,0]
    LOC_yn=[1,1,0,0,1,1,0,0]
    race=[1,0,1,0,0,1,0,1]
    means = pd.DataFrame({'tx_condition_pre': tx_condition,'LOC_YN_pre': LOC_yn,'Nonwhite_pre':race})
    
    #Get grand mean for other regressors.
    predict_df=df_testmeal_raw_wide
    predict_df=predict_df.drop(columns= ['tx_condition_pre', 'Nonwhite_pre','LOC_YN_pre',
           'TOTAL_CAL_CONSUME_post', 
           'TOTAL_PRO_PCT_CONSUME_ARCSIN_post', 
           'TOTAL_FAT_PCT_CONSUME_ARCSIN_post',
           'TOTAL_CARB_PCT_CONSUME_ARCSIN_post'])
    cov_means=statsmodels.stats.descriptivestats.describe(data=predict_df, stats=["mean"], numeric=True,categorical=True,alpha=0.05, use_t=False, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99), ntop=5)
    predictions=means.merge(cov_means, how='cross', validate="m:1")
    
    prediction_total_cal=(total_cal.predict(exog=predictions))
    prediction_summary=pd.DataFrame(prediction_total_cal)            
    prediction_summary=prediction_summary.rename(columns={0:'predict_value'})
    emm=pd.concat([predictions,prediction_summary], axis =1)
    emm=emm.drop(columns=['FAT_PERCENT_ARCSIN_pre', 'TOT_LEAN_pre', 'age_pre','height_pre', 'Nonwhite_pre','LOC_YN_pre'])
    
    emm_condition=emm.groupby(by="tx_condition_pre").mean() #EMM for condition comparisons
    emm_condition=emm_condition.reset_index()
    emm_condition=emm_condition.transpose()
    emm_condition=emm_condition.rename(columns={0:'emm_control',1:'emm_active'})
    emm_condition=emm_condition.drop(labels=['tx_condition_pre'],axis=0)
    emm_condition=emm_condition.rename(index={"predict_value":'C(tx_condition_pre)[T.1]'})
    emm_condition['mean_diff']=(emm_condition['emm_active']-emm_condition['emm_control'])
    
    #cohens d and 95% CI computations
    condition_summary=mixed_summary.drop(labels=['Intercept','C(Nonwhite_pre)[T.1]','FAT_PERCENT_ARCSIN_pre','age_pre','height_pre', 'C(LOC_YN_pre)[T.1]', 'TOT_LEAN_pre'],axis=0)
    condition_summary = pd.concat([condition_summary,emm_condition],axis=1)
    condition_summary['SD_pooled']= SD_pooled #SD is the pooled within-group standard deviation of the outcome measure (Y).
    condition_summary['SE_pooled']=condition_summary['SD_pooled']*(np.sqrt((1/n_ctrl)+(1/n_active)))
    condition_summary['cohens_d']=condition_summary['mean_diff']/condition_summary['SD_pooled'] #b is the unstandardized coefficient for the effect of group
    condition_summary['95CI']=1.96*condition_summary['SE_pooled'] 
    condition_summary['95CI_low_cohens_d']= condition_summary['cohens_d']-condition_summary['95CI']
    condition_summary['95CI_up_cohens_d']=condition_summary['cohens_d']+condition_summary['95CI'] 
    condition_summary['95CI_low_emm_ctrl']= condition_summary['emm_control']-(1.96*(condition_summary['SD_pooled']/(np.sqrt(n_ctrl))))
    condition_summary['95CI_up_emm_ctrl']=condition_summary['emm_control']+(1.96*(condition_summary['SD_pooled']/(np.sqrt(n_ctrl))))
    condition_summary['95CI_low_emm_active']= condition_summary['emm_active']-(1.96*(condition_summary['SD_pooled']/(np.sqrt(n_active))))
    condition_summary['95CI_up_emm_active']=condition_summary['emm_active']+(1.96*(condition_summary['SD_pooled']/(np.sqrt(n_active))))
    
    #create final output files
    condition_summary=condition_summary.reset_index()
    condition_summary=condition_summary.drop(columns=['index'])
    print (condition_summary['mean_diff'])
    print(condition_summary['beta']) # should be equal to emm difference score. 
    condition_summary.to_csv(f'{dir_base}/{today}_testmeal_ols_condition_{dv}.csv',sep=',', index=False)

            
#%% RUN General linear MODEL TESTING THE INTERACTION EFFECT OF CONDITION BY LOC-EATING ON CHANGE IN test meal intake    
def run_ols_loc_moderation (dv):
    
    interaction_model_total_cal = f"{dv}_post ~ {dv}_pre+  C(tx_condition_pre)*C(LOC_YN_pre)+ C(Nonwhite_pre) +TOT_LEAN_pre +FAT_PERCENT_ARCSIN_pre +age_pre +height_pre"
    total_cal_interaction = smf.ols(interaction_model_total_cal,data=df_testmeal_raw_wide).fit()
    total_cal_interaction.summary() 

    #save paramater estimates.        
    loc_param=pd.DataFrame(total_cal_interaction.params) #model parameters (betas)
    loc_param=loc_param.rename(columns={0: "beta"})
    loc_bse=pd.DataFrame(total_cal_interaction.bse) #standard errors of the parameter estimates.
    loc_bse=loc_bse.rename(columns={0: "se"})
    loc_ci=pd.DataFrame(total_cal_interaction.conf_int()) #parameter confidence intervals
    loc_ci=loc_ci.rename(columns={0: "95_lower",1:'95_upper'})
    loc_mixed_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
    loc_mixed_summary['p']=total_cal_interaction.pvalues
    
    #compute SD for each tx group based on residuals from the model. 
    resid=pd.DataFrame(total_cal_interaction.resid)
    resid=resid.rename(columns={0:'resid'})
    resid['resid_sq']=resid['resid']**2
    temp=pd.concat([resid,df_testmeal_raw_wide], axis =1)
          
    #get SD for condition main effects. 
    temp_condition=temp.pivot_table(index=['tx_condition_pre'],values=['resid','resid_sq'],aggfunc='sum')
    temp2_condition=temp.pivot_table(index=['tx_condition_pre'],aggfunc='size')
    #temp2_condition=temp2_condition.rename_axis('subj_count')
    temp_condition=pd.concat([temp_condition,temp2_condition], axis =1)
    temp_condition=temp_condition.rename(columns={0:'subj_count'})
    n_ctrl=temp_condition['subj_count'].loc[0]#control
    n_active=temp_condition['subj_count'].loc[1]#active
    temp_condition['var']=temp_condition['resid_sq']/(temp_condition['subj_count']-1)
    temp_condition['SD']=np.sqrt(temp_condition['var'])
    temp_condition['SD_sq']=(temp_condition['SD']**2)
    SD_ctrl=temp_condition['SD_sq'].loc[0]#control
    SD_active=temp_condition['SD_sq'].loc[1]#active   
    SD_pooled= math.sqrt((((n_ctrl-1)*SD_ctrl)+((n_active-1)*SD_active))/(n_ctrl+n_active-2))
      
    #get emm for condition 
    loc_prediction_model=(total_cal_interaction.predict(exog=predictions))
    loc_prediction_summary=pd.DataFrame(loc_prediction_model)            
    loc_prediction_summary=loc_prediction_summary.rename(columns={0:'predict_value'})
    
    loc_emm=pd.concat([predictions,loc_prediction_summary], axis =1)
    loc_emm_condition=loc_emm.groupby(by="tx_condition_pre").mean() #EMM for condition comparisons
    loc_emm_condition=loc_emm_condition.drop(columns=['FAT_PERCENT_ARCSIN_pre', 'TOT_LEAN_pre', 'age_pre','height_pre', 'Nonwhite_pre','LOC_YN_pre'])
    loc_emm_condition=loc_emm_condition.reset_index()
    loc_emm_condition=loc_emm_condition.transpose()
    loc_emm_condition=loc_emm_condition.rename(columns={0:'emm_control',1:'emm_active'})
    loc_emm_condition=loc_emm_condition.drop(labels=['tx_condition_pre'],axis=0)
    loc_emm_condition=loc_emm_condition.rename(index={"predict_value":'C(tx_condition_pre)[T.1]'})
    loc_emm_condition['mean_diff']=loc_emm_condition['emm_active']-loc_emm_condition['emm_control']
    
    #cohen's d and 95% CI computations for condition 
    loc_condition_summary=loc_mixed_summary.drop(labels=['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','C(LOC_YN_pre)[T.1]','Intercept','C(Nonwhite_pre)[T.1]','FAT_PERCENT_ARCSIN_pre','age_pre','height_pre','TOT_LEAN_pre'],axis=0)
    loc_condition_summary = pd.concat([loc_condition_summary,loc_emm_condition],axis=1)
    loc_condition_summary['SD_pooled']= SD_pooled #SD is the pooled within-group standard deviation of the outcome measure (Y).
    loc_condition_summary['SE_pooled']=loc_condition_summary['SD_pooled']*(np.sqrt((1/n_ctrl)+(1/n_active)))
    loc_condition_summary['cohens_d']=loc_condition_summary['mean_diff']/loc_condition_summary['SD_pooled'] #b is the unstandardized coefficient for the effect of group
    loc_condition_summary['95CI']=1.96*loc_condition_summary['SE_pooled'] 
    loc_condition_summary['95CI_low_cohens_d']= loc_condition_summary['cohens_d']-loc_condition_summary['95CI']
    loc_condition_summary['95CI_up_cohens_d']=loc_condition_summary['cohens_d']+loc_condition_summary['95CI'] 
    loc_condition_summary['95CI_low_emm_ctrl']= loc_condition_summary['emm_control']-(1.96*(loc_condition_summary['SD_pooled']/(np.sqrt(n_ctrl))))
    loc_condition_summary['95CI_up_emm_ctrl']=loc_condition_summary['emm_control']+(1.96*(loc_condition_summary['SD_pooled']/(np.sqrt(n_ctrl))))
    loc_condition_summary['95CI_low_emm_active']= loc_condition_summary['emm_active']-(1.96*(loc_condition_summary['SD_pooled']/(np.sqrt(n_active))))
    loc_condition_summary['95CI_up_emm_active']=loc_condition_summary['emm_active']+(1.96*(loc_condition_summary['SD_pooled']/(np.sqrt(n_active))))
    
    loc_condition_summary=loc_condition_summary.reset_index()
    loc_condition_summary=loc_condition_summary.drop(columns=['index'])
    loc_condition_summary.to_csv(f'{dir_base}/{today}_testmeal_ols_loc_condition_{dv}.csv',sep=',', index=False)
    
    ############################
    #Interaction effect size . 
    ###########################
     #get SD for condition by LOC interaction effects. 
    temp_loc=temp.pivot_table(index=['tx_condition_pre','LOC_YN_pre'],values=['resid','resid_sq'],aggfunc='sum')
    temp2_loc=temp.pivot_table(index=['tx_condition_pre','LOC_YN_pre'],values=['resid','resid_sq'],aggfunc='size')
    temp_loc=pd.concat([temp_loc,temp2_loc], axis =1)
    temp_loc=temp_loc.rename(columns={0:'subj_count'})
    n_CnoL = temp_loc.loc[(0, 0), 'subj_count']#control no loc
    n_CL = temp_loc.loc[(0, 1), 'subj_count']#control with loc
    n_AnoL = temp_loc.loc[(1, 0), 'subj_count'] #active no loc
    n_AL=temp_loc.loc[(1, 1), 'subj_count'] #active with loc
     
    temp_loc['var']=temp_loc['resid_sq']/(temp_loc['subj_count']-1)
    temp_loc['SD']=np.sqrt(temp_loc['var'])
    temp_loc['SD_sq']=(temp_loc['SD']**2)
    SD_CnoL=temp_loc.loc[(0, 0),'SD_sq'] #control no loc
    SD_CL=temp_loc.loc[(0,1), 'SD_sq'] #control with loc
    SD_AnoL=temp_loc.loc[(1, 0), 'SD_sq']#active no loc
    SD_AL=temp_loc.loc[(1, 1), 'SD_sq']#active with loc
         
    #entire sample, 4 group pooled SD.
    SD_pool_numerator= (((n_CnoL-1)*SD_CnoL)+((n_AnoL-1)*SD_AnoL)+((n_CL-1)*SD_CL)+((n_AL-1)*SD_AL))
    SD_pool_denominator=(n_CnoL+n_AnoL+n_CL+n_AL-4)
    SD_pooled=np.sqrt(SD_pool_numerator/SD_pool_denominator)    
        
    #get emm for condition  by LOC 
    loc_emm=pd.concat([predictions,loc_prediction_summary], axis =1)
    loc_emm_LOC_interaction=loc_emm.groupby(by=["tx_condition_pre","LOC_YN_pre"]).mean() #EMM for LOC by condition comparisons
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['FAT_PERCENT_ARCSIN_pre', 'TOT_LEAN_pre', 'age_pre','height_pre', 'Nonwhite_pre'])
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.transpose()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(columns={0:'emm_control_noloc',1:'emm_control_loc',2:'emm_active_noloc',3:'emm_active_loc'})
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(labels=['tx_condition_pre','LOC_YN_pre'],axis=0)
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(index={"predict_value":'C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]'})
    
    emm_CnoL= loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']
    emm_CL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']
    emm_AnoL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']
    emm_AL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    
    #estimate 95%CI for emm for each condition.
    CnoL_emm_95CI_low= emm_CnoL-(1.96*(SD_pooled/(np.sqrt(n_CnoL))))
    CnoL_emm_95CI_high= emm_CnoL+(1.96*(SD_pooled/(np.sqrt(n_CnoL))))
   
    CL_emm_95CI_low= emm_CL-(1.96*(SD_pooled/(np.sqrt(n_CL))))
    CL_emm_95CI_high= emm_CL+(1.96*(SD_pooled/(np.sqrt(n_CL))))

    AnoL_emm_95CI_low= emm_AnoL-(1.96*(SD_pooled/(np.sqrt(n_AnoL))))
    AnoL_emm_95CI_high= emm_AnoL+(1.96*(SD_pooled/(np.sqrt(n_AnoL))))

    AL_emm_95CI_low= emm_AL-(1.96*(SD_pooled/(np.sqrt(n_AL))))
    AL_emm_95CI_high= emm_AL+(1.96*(SD_pooled/(np.sqrt(n_AL))))

    #write emm and 95% to cvs.
    LOC_interaction_emm_95CI={'group':['CnoL','CL','AnoL','AL'],
                      'emm':[emm_CnoL,emm_CL,emm_AnoL,emm_AL],
                      "95%CI_high":[CnoL_emm_95CI_high,CL_emm_95CI_high,AnoL_emm_95CI_high,AL_emm_95CI_high],
                      "95%CI_low":[CnoL_emm_95CI_low,CL_emm_95CI_low,AnoL_emm_95CI_low,AL_emm_95CI_low]}
    emm95_dframe=pd.DataFrame.from_dict(LOC_interaction_emm_95CI)

    emm95_dframe.to_csv(f'{dir_base}/{today}_testmeal_ols_emm95_interaction_{dv}.csv',sep=',', index=False)

   
    #estimate cohens d for each comparison
    mean_diff_CnoL_CL= loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']
    mean_diff_CnoL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']
    mean_diff_CnoL_A=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    mean_diff_CL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']
    mean_diff_CL_AL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    mean_diff_AnoL_AL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
   
    SD_CnoL_CL= math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_CL-1)*SD_CL))/(n_CnoL+n_CL-2))
    SD_CnoL_AnoL=math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_AnoL-1)*SD_AnoL))/(n_CnoL+n_AnoL-2))
    SD_CnoL_AL=math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_AL-1)*SD_AL))/(n_CnoL+n_AL-2))
    SD_CL_AnoL=math.sqrt((((n_CL-1)*SD_CL)+((n_AnoL-1)*SD_AnoL))/(n_CL+n_AnoL-2))
    SD_CL_AL=math.sqrt((((n_CL-1)*SD_CL)+((n_AL-1)*SD_AL))/(n_CL+n_AL-2))
    SD_AnoL_AL=math.sqrt((((n_AnoL-1)*SD_AnoL)+((n_AL-1)*SD_AL))/(n_AnoL+n_AL-2))
   
    SE_CnoL_CL=SD_CnoL_CL*(np.sqrt((1/n_CnoL)+(1/n_CL)))
    SE_CnoL_AnoL=SD_CnoL_AnoL*(np.sqrt((1/n_CnoL)+(1/n_AnoL)))
    SE_CnoL_AL=SD_CnoL_AL*(np.sqrt((1/n_CnoL)+(1/n_AL)))
    SE_CL_AnoL=SD_CL_AnoL*(np.sqrt((1/n_CL )+(1/n_AnoL)))
    SE_CL_AL=SD_CL_AL*(np.sqrt((1/n_CL )+(1/n_AL)))
    SE_AnoL_AL=SD_AnoL_AL*(np.sqrt((1/ n_AnoL)+(1/n_AL)))
     
    loc_emm_LOC_interaction
    interaction_data={'comparison':['CnoL_CL','CnoL_AnoL','CnoL_AL','CL_AnoL','CL_AL','AnoL_AL'],
                      'mean_diff':[mean_diff_CnoL_CL,mean_diff_CnoL_AnoL,mean_diff_CnoL_A,mean_diff_CL_AnoL,mean_diff_CL_AL,mean_diff_AnoL_AL],
                      "SD":[SD_CnoL_CL,SD_CnoL_AnoL,SD_CnoL_AL,SD_CL_AnoL,SD_CL_AL,SD_AnoL_AL],
                      "SE_pooled": [SE_CnoL_CL,SE_CnoL_AnoL,SE_CnoL_AL,SE_CL_AnoL,SE_CL_AL,SE_AnoL_AL]}
    interaction_dframe=pd.DataFrame.from_dict(interaction_data)
    interaction_dframe['cohens_d']=interaction_dframe['mean_diff']/interaction_dframe['SD']
    interaction_dframe['95CI']=1.96*interaction_dframe['SE_pooled']
    interaction_dframe['95CI_low_cohens_d']= interaction_dframe['cohens_d']-interaction_dframe['95CI']
    interaction_dframe['95CI_up_cohens_d']=interaction_dframe['cohens_d']+interaction_dframe['95CI']
    

    #estimate cohens d for each comparison
    mean_diff_CnoL_CL= loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']
    mean_diff_CnoL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']
    mean_diff_CnoL_A=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    mean_diff_CL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']
    mean_diff_CL_AL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    mean_diff_AnoL_AL=loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_noloc']-loc_emm_LOC_interaction.at['C(tx_condition_pre)[T.1]:C(LOC_YN_pre)[T.1]','emm_active_loc']
    
    SD_CnoL_CL= math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_CL-1)*SD_CL))/(n_CnoL+n_CL-2))
    SD_CnoL_AnoL=math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_AnoL-1)*SD_AnoL))/(n_CnoL+n_AnoL-2))
    SD_CnoL_AL=math.sqrt((((n_CnoL-1)*SD_CnoL)+((n_AL-1)*SD_AL))/(n_CnoL+n_AL-2))
    SD_CL_AnoL=math.sqrt((((n_CL-1)*SD_CL)+((n_AnoL-1)*SD_AnoL))/(n_CL+n_AnoL-2))
    SD_CL_AL=math.sqrt((((n_CL-1)*SD_CL)+((n_AL-1)*SD_AL))/(n_CL+n_AL-2))
    SD_AnoL_AL=math.sqrt((((n_AnoL-1)*SD_AnoL)+((n_AL-1)*SD_AL))/(n_AnoL+n_AL-2))
    
    SE_CnoL_CL=SD_CnoL_CL*(np.sqrt((1/n_CnoL)+(1/n_CL)))
    SE_CnoL_AnoL=SD_CnoL_AnoL*(np.sqrt((1/n_CnoL)+(1/n_AnoL)))
    SE_CnoL_AL=SD_CnoL_AL*(np.sqrt((1/n_CnoL)+(1/n_AL)))
    SE_CL_AnoL=SD_CL_AnoL*(np.sqrt((1/n_CL )+(1/n_AnoL)))
    SE_CL_AL=SD_CL_AL*(np.sqrt((1/n_CL )+(1/n_AL)))
    SE_AnoL_AL=SD_AnoL_AL*(np.sqrt((1/ n_AnoL)+(1/n_AL)))
     
    
    interaction_data={'comparison':['CnoL_CL','CnoL_AnoL','CnoL_AL','CL_AnoL','CL_AL','AnoL_AL'],
                      'mean_diff':[mean_diff_CnoL_CL,mean_diff_CnoL_AnoL,mean_diff_CnoL_A,mean_diff_CL_AnoL,mean_diff_CL_AL,mean_diff_AnoL_AL],
                      "SD":[SD_CnoL_CL,SD_CnoL_AnoL,SD_CnoL_AL,SD_CL_AnoL,SD_CL_AL,SD_AnoL_AL],
                      "SE_pooled": [SE_CnoL_CL,SE_CnoL_AnoL,SE_CnoL_AL,SE_CL_AnoL,SE_CL_AL,SE_AnoL_AL]}
    interaction_dframe=pd.DataFrame.from_dict(interaction_data)
    interaction_dframe['cohens_d']=interaction_dframe['mean_diff']/interaction_dframe['SD']
    interaction_dframe['95CI']=1.96*interaction_dframe['SE_pooled'] 
    interaction_dframe['95CI_low_traditional']= interaction_dframe['cohens_d']-interaction_dframe['95CI']
    interaction_dframe['95CI_up_traditional']=interaction_dframe['cohens_d']+interaction_dframe['95CI'] 
    
    interaction_dframe.to_csv(f'{dir_base}/{today}_testmeal_ols_loc_{dv}.csv',sep=',', index=False)
    
    ############################
    #Interaction estimates. 
    ###########################
    #save model params for interaction 
    loc_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
    loc_summary['p']=total_cal_interaction.pvalues
    loc_summary=loc_summary.drop(labels=['C(tx_condition_pre)[T.1]','C(LOC_YN_pre)[T.1]','Intercept','C(Nonwhite_pre)[T.1]','FAT_PERCENT_ARCSIN_pre','age_pre','height_pre', 'TOT_LEAN_pre'],axis=0)
    loc_summary=loc_summary.reset_index()
    loc_summary=loc_summary.drop(columns=['index'])  
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['index'])
    loc_summary=pd.concat([loc_summary,loc_emm_LOC_interaction], axis =1)
    
    loc_summary.to_csv(f'{dir_base}/{today}_testmeal_ols_LOCinteraction_{dv}.csv',sep=',', index=False)


#%% Loop over different test meal metrics to obtain results for all outcomes

dv_list=["TOTAL_CAL_CONSUME", "TOTAL_PRO_PCT_CONSUME_ARCSIN", "TOTAL_FAT_PCT_CONSUME_ARCSIN", "TOTAL_CARB_PCT_CONSUME_ARCSIN"]
for dv in dv_list:
    run_ols(dv)      
    run_ols_loc_moderation(dv)




