#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 20 12:10:18 2023

@author: MegNParker

"""
#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels as statsmodels
import os, os.path as op
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import math

#%% IMPORT DATAFILES

#Navigate to the folder where data are located.
dir_base= "..." #directory where you saved combined sct files for group level statistics. "outname_ts" in separated_mixed_source_estimates.py
os.chdir(dir_base) 

#read in meg data.
meg_dset=pd.read_csv('meg_dset.csv',sep=',')
meg_dset.head

#check number of included sub
subj_count=meg_dset.pivot_table(index=['subjid'],aggfunc='size')
subj_count.shape[0] 

sub_condition=meg_dset.pivot_table(index=['subjid','tx_condition'],aggfunc='size')
sub_condition=sub_condition.reset_index()
sub_condition=sub_condition['tx_condition'].value_counts()

# visualize raw data
meg_dset['ts_change'].max()
meg_dset['ts_change'].min()
meg_dset['ts_change'].skew()
meg_dset['ts_change'].kurtosis()

#%% prediction dset.

#list of possible comparison, condition and LOC combos. 
comparison=[1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
condition_coded=[1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]
LOC_yn=[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
race=[1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0]
means = pd.DataFrame({'comparison': comparison,'tx_condition': condition_coded,'LOC_YN': LOC_yn,'Nonwhite':race})
    
#Get grand mean for other regressors. 
predict_df=meg_dset
predict_df=predict_df.drop(columns=[ 'ROI', 'timeframe', 'ts_pre', 'ts_post', 'ts_change','comparison','LOC_YN','tx_condition','Nonwhite'])
predict_df=predict_df.drop_duplicates(subset='subjid')
cov_means=statsmodels.stats.descriptivestats.describe(data=predict_df, stats=["mean"], numeric=True,categorical=True,alpha=0.05, use_t=False, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99), ntop=5)
predictions=means.merge(cov_means, how='cross', validate="m:1")

#%% CREATE NECESARY VARS, LISTS, DATAFRAMES

#list of ROIs you want to use in analyses. 
ROI_list=["caudalanteriorcingulate-lh",
"caudalanteriorcingulate-rh",
"caudalmiddlefrontal-lh",
"caudalmiddlefrontal-rh",
"lateralorbitofrontal-lh",
"lateralorbitofrontal-rh",
"Left-Caudate",
"Left-Pallidum",
"Left-Putamen",
"medialorbitofrontal-lh",
"medialorbitofrontal-rh",
"parsopercularis-lh",
"parsopercularis-rh",
"parsorbitalis-lh",
"parsorbitalis-rh",
"parstriangularis-lh",
"parstriangularis-rh",
"Right-Caudate",
"Right-Pallidum",
"Right-Putamen",
"rostralanteriorcingulate-lh",
"rostralanteriorcingulate-rh",
"rostralmiddlefrontal-lh",
"rostralmiddlefrontal-rh",
"superiorfrontal-lh",
"superiorfrontal-rh"]

#list of distinct timeframes you want to use in the analyses. 
timeframes=["ad","ucr"]
f_dframe=pd.DataFrame(columns=['sum_sq','df','F','PR(>F)','ROI','time'])

#for initializing the code. 
ROI='Right-Pallidum'
timeframe='ad'
i='parsopercularis-lh'
x='ucr'
data=meg_dset.loc[(meg_dset['ROI'] == i) & (meg_dset['timeframe'] == x)]

#%%Create directories for output files

# Import date class from datetime module
from datetime import date
today = date.today()

# Create new directory for masater dataframe outputs
out_dir_name=f"{dir_base}/masterdata_outputs/{today}"
if not os.path.exists(out_dir_name):
    os.mkdir(out_dir_name)
    
os.chdir(out_dir_name)
out_dir=os.curdir
out_fname=op.join(out_dir,f"masterdset_{today}.csv")
meg_dset.to_csv(out_fname)

# Create new directory for individual ROI/timeframe outputs
dir_name=f'{dir_base}/ROI_summaries/{today}'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

#%% RUN MIXED MODEL TESTING THE EFFECT OF CONDITION ON CHANGE IN OSCILATORY POWER - FOR 1 ROI AND 1 TIMEFRAME, AND OBTAIN EMMS

def run_mlm(data):
    #run model
    model = sm.MixedLM.from_formula('ts_change ~ C(tx_condition)+ C(Nonwhite) +TOT_FAT +age +height +C(comparison)',data=data,groups=data["subjid"]).fit()
    model.summary()

    param=pd.DataFrame(model.fe_params)
    param=param.rename(columns={0: "beta"})
    bse=pd.DataFrame(model.bse_fe)
    bse=bse.rename(columns={0: "se"})
    ci=pd.DataFrame(model.conf_int())
    ci=ci.rename(columns={0: "95_lower",1:'95_upper'})
    mixed_summary = pd.concat([param, bse,ci],axis=1)
    mixed_summary['ROI']=ROI
    mixed_summary['time']=timeframe
    mixed_summary['p']=model.pvalues
    
    #compute SD for each tx group based on residuals from the model. 
    resid=pd.DataFrame(model.resid)
    resid=resid.rename(columns={0:'resid'})
    resid['resid_sq']=resid['resid']**2
    temp=pd.concat([resid,data], axis =1)
    temp1=temp.pivot_table(index=['tx_condition'],values=['resid','resid_sq'],aggfunc='sum')
    temp2=temp.pivot_table(index=['tx_condition'],values=['resid','resid_sq'],aggfunc='size')
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
    
    prediction_model=(model.predict(exog=predictions))
    prediction_summary=pd.DataFrame(prediction_model)            
    prediction_summary=prediction_summary.rename(columns={0:'predict_value'})
    emm=pd.concat([predictions,prediction_summary], axis =1)
    emm=emm.drop(columns=['comparison', 'TOT_FAT', 'age','height', 'Nonwhite','LOC_YN'])

    emm_condition=emm.groupby(by="tx_condition").mean() #EMM for condition comparisons
    emm_condition=emm_condition.reset_index()
    emm_condition=emm_condition.transpose()
    emm_condition=emm_condition.rename(columns={0:'emm_control',1:'emm_active'})
    emm_condition=emm_condition.drop(labels=['tx_condition'],axis=0)
    emm_condition=emm_condition.rename(index={"predict_value":'C(tx_condition)[T.1]'})
    emm_condition['mean_diff']=(emm_condition['emm_active']-emm_condition['emm_control'])
    
    #traditional d computations
    condition_summary=mixed_summary.drop(labels=['Intercept','C(Nonwhite)[T.1]','C(comparison)[T.2]','C(comparison)[T.3]','TOT_FAT','age','height','Group Var'],axis=0)
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
    condition_summary.to_csv(f'{dir_base}/ROI_summaries/{today}/{ROI}_{timeframe}_{today}_mlm_condition.csv',sep=',', index=False)

            
#%% RUN MIXED MODEL TESTING THE INTERACTION EFFECT OF CONDITION BY LOC-EATING ON CHANGE IN OSCILATORY POWER - FOR 1 ROI AND 1 TIMEFRAME, AND OBTAIN EMMS

def run_mlm_loc_moderation(data):
            
    #run model
    loc_model = sm.MixedLM.from_formula('ts_change ~ C(tx_condition)*C(LOC_YN)+ C(Nonwhite) +TOT_FAT +age +height +C(comparison)',data=data,groups=data["subjid"]).fit()
    loc_model.summary()

    #save paramater estimates.        
    
    loc_param=pd.DataFrame(loc_model.fe_params) #model parameters (betas)
    loc_param=loc_param.rename(columns={0: "beta"})
    loc_bse=pd.DataFrame(loc_model.bse_fe) #standard errors of the parameter estimates.
    loc_bse=loc_bse.rename(columns={0: "se"})
    loc_ci=pd.DataFrame(loc_model.conf_int()) #parameter confidence intervals
    loc_ci=loc_ci.rename(columns={0: "95_lower",1:'95_upper'})
    loc_mixed_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
    loc_mixed_summary['ROI']=ROI
    loc_mixed_summary['time']=timeframe
    loc_mixed_summary['p']=loc_model.pvalues
    
    
    #compute SD for each tx group based on residuals from the model. 
    resid=pd.DataFrame(loc_model.resid)
    resid=resid.rename(columns={0:'resid'})
    resid['resid_sq']=resid['resid']**2
    temp=pd.concat([resid,data], axis =1)
          
    #get SD for condition main effects. 
    temp_condition=temp.pivot_table(index=['tx_condition'],values=['resid','resid_sq'],aggfunc='sum')
    temp2_condition=temp.pivot_table(index=['tx_condition'],aggfunc='size')
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
    loc_prediction_model=(loc_model.predict(exog=predictions))
    loc_prediction_summary=pd.DataFrame(loc_prediction_model)            
    loc_prediction_summary=loc_prediction_summary.rename(columns={0:'predict_value'})
    
    loc_emm=pd.concat([predictions,loc_prediction_summary], axis =1)
    loc_emm_condition=loc_emm.groupby(by="tx_condition").mean() #EMM for condition comparisons
    loc_emm_condition=loc_emm_condition.drop(columns=['comparison', 'TOT_FAT', 'age','height', 'Nonwhite','LOC_YN'])
    loc_emm_condition=loc_emm_condition.reset_index()
    loc_emm_condition=loc_emm_condition.transpose()
    loc_emm_condition=loc_emm_condition.rename(columns={0:'emm_control',1:'emm_active'})
    loc_emm_condition=loc_emm_condition.drop(labels=['tx_condition'],axis=0)
    loc_emm_condition=loc_emm_condition.rename(index={"predict_value":'C(tx_condition)[T.1]'})
    loc_emm_condition['mean_diff']=loc_emm_condition['emm_active']-loc_emm_condition['emm_control']
    
    #traditional d computations for condition 
    loc_condition_summary=loc_mixed_summary.drop(labels=['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','C(LOC_YN)[T.1]','Intercept','C(Nonwhite)[T.1]','C(comparison)[T.2]','C(comparison)[T.3]','TOT_FAT','age','height','Group Var'],axis=0)
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
    
    loc_condition_summary.to_csv(f'{dir_base}/ROI_summaries/{today}/{ROI}_{timeframe}_{today}_mlm_LOCcondition.csv',sep=',', index=False)

    ############################
    #Interaction effect size . 
    ###########################
     #get SD for condition by LOC interaction effects. 
    temp_loc=temp.pivot_table(index=['tx_condition','LOC_YN'],values=['resid','resid_sq'],aggfunc='sum')
    temp2_loc=temp.pivot_table(index=['tx_condition','LOC_YN'],values=['resid','resid_sq'],aggfunc='size')
    temp_loc=pd.concat([temp_loc,temp2_loc], axis =1)
    temp_loc=temp_loc.rename(columns={0:'subj_count'})
    n_CnoL=temp_loc['subj_count'].loc[0,0]#control no loc
    n_CL=temp_loc['subj_count'].loc[0,1]#control with loc
    n_AnoL=temp_loc['subj_count'].loc[1,0]#active no loc
    n_AL=temp_loc['subj_count'].loc[1,1]#active with loc
     
    temp_loc['var']=temp_loc['resid_sq']/(temp_loc['subj_count']-1)
    temp_loc['SD']=np.sqrt(temp_loc['var'])
    temp_loc['SD_sq']=(temp_loc['SD']**2)
    SD_CnoL=temp_loc['SD_sq'].loc[0,0]#control no loc
    SD_CL=temp_loc['SD_sq'].loc[0,1]#control with loc
    SD_AnoL=temp_loc['SD_sq'].loc[1,0]#active no loc
    SD_AL=temp_loc['SD_sq'].loc[1,1]#active with loc

    #entire sample, 4 group pooled SD. 
    SD_pool_numerator= (((n_CnoL-1)*SD_CnoL)+((n_AnoL-1)*SD_AnoL)+((n_CL-1)*SD_CL)+((n_AL-1)*SD_AL))
    SD_pool_denominator=(n_CnoL+n_AnoL+n_CL+n_AL-4)
    SD_pooled=np.sqrt(SD_pool_numerator/SD_pool_denominator) 


    #get emm for condition  by LOC 
    loc_emm=pd.concat([predictions,loc_prediction_summary], axis =1)
    loc_emm_LOC_interaction=loc_emm.groupby(by=["tx_condition","LOC_YN"]).mean() #EMM for LOC by condition comparisons
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['comparison', 'TOT_FAT', 'age','height', 'Nonwhite'])
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.transpose()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(columns={0:'emm_control_noloc',1:'emm_control_loc',2:'emm_active_noloc',3:'emm_active_loc'})
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(labels=['tx_condition','LOC_YN'],axis=0)
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(index={"predict_value":'C(tx_condition)[T.1]:C(LOC_YN)[T.1]'})

    emm_CnoL= loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_noloc']
    emm_CL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_loc']
    emm_AnoL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_noloc']
    emm_AL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_loc']


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
    emm95_dframe['ROI']=ROI
    emm95_dframe['time']=timeframe
    emm95_dframe.to_csv(f'{dir_base}/yanovski_meg/Longitudinal_analysis/data_analysis_scripts/ROI_summaries/{today}/{ROI}_{timeframe}_{today}_mlm_LOCinteraction_emm95CI.csv',sep=',', index=False)

    
    #estimate cohens d for each comparison
    mean_diff_CnoL_CL= loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_loc']
    mean_diff_CnoL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_noloc']
    mean_diff_CnoL_A=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_noloc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_loc']
    mean_diff_CL_AnoL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_noloc']
    mean_diff_CL_AL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_control_loc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_loc']
    mean_diff_AnoL_AL=loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_noloc']-loc_emm_LOC_interaction.at['C(tx_condition)[T.1]:C(LOC_YN)[T.1]','emm_active_loc']
    
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
    interaction_dframe['ROI']=ROI
    interaction_dframe['time']=timeframe


    #print(loc_condition_summary)
    interaction_dframe.to_csv(f'{dir_base}/ROI_summaries/{today}/{ROI}_{timeframe}_{today}_mlm_LOCinteraction_es.csv',sep=',', index=False)
    
    ############################
    #Interaction estimates. 
    ###########################
    #save model params for interaction 
    loc_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
    loc_summary['ROI']=ROI
    loc_summary['time']=timeframe
    loc_summary['p']=loc_model.pvalues
    loc_summary=loc_summary.drop(labels=['C(tx_condition)[T.1]','C(LOC_YN)[T.1]','Intercept','C(Nonwhite)[T.1]','C(comparison)[T.2]','C(comparison)[T.3]','TOT_FAT','age','height','Group Var'],axis=0)
    loc_summary=loc_summary.reset_index()
    loc_summary=loc_summary.drop(columns=['index'])  
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
    loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['index'])
    loc_summary=pd.concat([loc_summary,loc_emm_LOC_interaction], axis =1)
    loc_summary.to_csv(f'{dir_base}/ROI_summaries/{today}/{ROI}_{timeframe}_{today}_mlm_LOCinteraction_params.csv',sep=',', index=False)

    
#%% LOOP OVER ALL ROIS AND TIMEFRAMES IN YOUR LISTS. 

for i in ROI_list:
    for x in timeframes:
        data_path=f"{dir_base}/masterdata_outputs/{today}/masterdset_{today}.csv"
        master_df=pd.DataFrame(pd.read_csv(data_path,index_col=False))
        data=master_df.loc[(master_df['ROI'] == i) & (master_df['timeframe'] == x)]
        ROI=i
        timeframe=x
        run_mlm(data)       
        run_mlm_loc_moderation(data)
        
        
#CREATE MASTER EXCEL FILE THEN COMBINE DATA FROM EACH ROI AND TIMEFRAME. 
os.chdir(dir_base+f"/ROI_summaries/{today}")

#conditon main effects model 
columns_condition=condition_summary.columns
condition_extension = 'mlm_condition.csv'
condition_list = glob.glob('*{}'.format(condition_extension))
outfile_condition=f"{dir_base}/masterdata_outputs/{today}/mlm_condition_MASTER_{today}.csv"
df_out=pd.DataFrame(columns=columns_condition)
df_out.to_csv(outfile_condition, index=False)

for file in condition_list:
    df=pd.read_csv(file,sep=",",index_col=False)
    master_df=pd.read_csv(outfile_condition,index_col=False)
    master_df=pd.concat([master_df,df])
    master_df.to_csv(outfile_condition, index=False)

#effects of conditon in interaction  model     
columns_loc_condition=loc_condition_summary.columns
loc_condition_extension = 'mlm_LOCcondition.csv'
loc_condition_list = glob.glob('*{}'.format(loc_condition_extension))
outfile_loc_condition=f"{dir_base}/masterdata_outputs/{today}/mlm_loc_condition_MASTER_{today}.csv"
df_out=pd.DataFrame(columns=columns_loc_condition)
df_out.to_csv(outfile_loc_condition, index=False)

for file in loc_condition_list:
    df=pd.read_csv(file,sep=",",index_col=False)
    master_df=pd.read_csv(outfile_loc_condition,index_col=False)
    master_df=pd.concat([master_df,df])
    master_df.to_csv(outfile_loc_condition, index=False)

# interaction pairwise effect sizes      
columns_interaction=interaction_dframe.columns
interaction_extension = 'mlm_LOCinteraction_es.csv'
interaction_list = glob.glob('*{}'.format(interaction_extension))
outfile_interaction=f"{dir_base}/masterdata_outputs/{today}/mlm_interaction_MASTER_{today}.csv"
df_out=pd.DataFrame(columns=columns_interaction)
df_out.to_csv(outfile_interaction, index=False)

for file in interaction_list:
    df=pd.read_csv(file,sep=",",index_col=False)
    master_df=pd.read_csv(outfile_interaction,index_col=False)
    master_df=pd.concat([master_df,df])
    master_df.to_csv(outfile_interaction, index=False)

#effects of  interaction       
columns_interact_param=loc_summary.columns
interact_param_extension = 'mlm_LOCinteraction_params.csv'
interact_param_list = glob.glob('*{}'.format(interact_param_extension))
outfile_interaction_param=f"{dir_base}/masterdata_outputs/{today}/mlm_interaction_param_MASTER_{today}.csv"
df_out=pd.DataFrame(columns=columns_interact_param)
df_out.to_csv(outfile_interaction_param, index=False)

for file in interact_param_list:
    df=pd.read_csv(file,sep=",",index_col=False)
    master_df=pd.read_csv(outfile_interaction_param,index_col=False)
    master_df=pd.concat([master_df,df])  
    master_df.to_csv(outfile_interaction_param, index=False)


#emm and 95%CI of  interaction       
columns_interact_emm=emm95_dframe.columns
interact_emm_extension = 'mlm_LOCinteraction_emm95CI.csv' 
interact_emm_list = glob.glob('*{}'.format(interact_emm_extension))
outfile_interaction_emm=f"{dir_base}/masterdata_outputs/{today}/mlm_interaction_emm_MASTER_{today}.csv"
df_out=pd.DataFrame(columns=columns_interact_emm)
df_out.to_csv(outfile_interaction_emm, index=False)

for file in interact_emm_list:
    df=pd.read_csv(file,sep=",",index_col=False)
    master_df=pd.read_csv(outfile_interaction_emm,index_col=False)
    master_df=pd.concat([master_df,df])
    master_df.to_csv(outfile_interaction_emm, index=False)
  







