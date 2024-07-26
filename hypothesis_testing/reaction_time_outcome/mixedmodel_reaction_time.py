
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 2024

@author: MegNParker

"""
#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels as statsmodels
import math
from scipy.stats import skew 
from scipy.stats import kurtosis

# Import date class from datetime module
from datetime import date
today = date.today()

#%% IMPORT DATAFILES
#Navigate to the folder where data are located. 
dir_base="..." #directory that contains test meal csv file. 

#read in csv data.
rxn_dset=pd.read_csv(f"{dir_base}/reaction_time_data.csv")

#check number of  sub
subj_count=rxn_dset.pivot_table(index=['subjid'],aggfunc='size')
subj_count.shape
num_obvs=rxn_dset.shape[0]
num_subs=subj_count.shape[0]

#check skewness and kurtosis. 
max1=rxn_dset['dotprobe_change'].max()
min1=rxn_dset['dotprobe_change'].min()
skew=skew(rxn_dset['dotprobe_change'])
kurtosis=kurtosis(rxn_dset['dotprobe_change'])

print(max1,min1)
print(skew,kurtosis)

max1=rxn_dset['dotprobe_pre'].max()
min1=rxn_dset['dotprobe_pre'].min()
skew=skew(rxn_dset['dotprobe_pre'])
kurtosis=kurtosis(rxn_dset['dotprobe_pre'])

print(max1,min1)
print(skew,kurtosis)

max1=rxn_dset['dotprobe_post'].max()
min1=rxn_dset['dotprobe_post'].min()
skew=skew(rxn_dset['dotprobe_post'])
kurtosis=kurtosis(rxn_dset['dotprobe_post'])

print(max1,min1)
print(skew,kurtosis)

#%% #prediction dset. not sure if need... 

#list of possible comparison, condition and LOC combos. 
comparison=[1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
condition_coded=[1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]
LOC_yn=[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
race=[1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0]
means = pd.DataFrame({'comparison': comparison,'tx_condition': condition_coded,'LOC_YN': LOC_yn,'Nonwhite':race})
    
#Get grand mean for other regressors. 
predict_df=rxn_dset
predict_df=predict_df.drop(columns=['dotprobe_pre', 'dotprobe_post', 'dotprobe_change','comparison','LOC_YN','tx_condition','Nonwhite'])
predict_df=predict_df.drop_duplicates(subset='subjid')
cov_means=statsmodels.stats.descriptivestats.describe(data=predict_df, stats=["mean"], numeric=True,categorical=True,alpha=0.05, use_t=False, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99), ntop=5)
predictions=means.merge(cov_means, how='cross', validate="m:1")

#%% RUN LINEAR MIXED MODEL TESTING THE EFFECT OF CONDITION ON CHANGE IN REACTION TIME AB SCORES 

#run model
model = sm.MixedLM.from_formula('dotprobe_change ~ C(tx_condition)+ C(Nonwhite) +TOT_FAT +age +height +C(comparison)',data=rxn_dset,groups=rxn_dset["subjid"]).fit()
model.summary()

#save paramater estimates.        
param=pd.DataFrame(model.fe_params)
param=param.rename(columns={0: "beta"})
bse=pd.DataFrame(model.bse_fe)
bse=bse.rename(columns={0: "se"})
ci=pd.DataFrame(model.conf_int())
ci=ci.rename(columns={0: "95_lower",1:'95_upper'})
mixed_summary = pd.concat([param, bse,ci],axis=1)
mixed_summary['p']=model.pvalues

#compute SD for each tx group based on residuals from the model. 
resid=pd.DataFrame(model.resid)
resid=resid.rename(columns={0:'resid'})
resid['resid_sq']=resid['resid']**2
temp=pd.concat([resid,rxn_dset], axis =1)
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
condition_summary.to_csv(f'{dir_base}/{today}_rxntime_mlm_condition.csv',sep=',', index=False)

            
#%% RUN MIXED MODEL TESTING THE INTERACTION EFFECT OF CONDITION BY LOC-EATING ON CHANGE IN REACTION TIME AV SCORES

#run model
loc_model = sm.MixedLM.from_formula('dotprobe_change ~ C(tx_condition)*C(LOC_YN)+ C(Nonwhite) +TOT_FAT +age +height +C(comparison)',data=rxn_dset,groups=rxn_dset["subjid"]).fit()
loc_model.summary()

#save paramater estimates.        
loc_param=pd.DataFrame(loc_model.fe_params) #model parameters (betas)
loc_param=loc_param.rename(columns={0: "beta"})
loc_bse=pd.DataFrame(loc_model.bse_fe) #standard errors of the parameter estimates.
loc_bse=loc_bse.rename(columns={0: "se"})
loc_ci=pd.DataFrame(loc_model.conf_int()) #parameter confidence intervals
loc_ci=loc_ci.rename(columns={0: "95_lower",1:'95_upper'})
loc_mixed_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
loc_mixed_summary['p']=loc_model.pvalues

#compute SD for each tx group based on residuals from the model. 
resid=pd.DataFrame(loc_model.resid)
resid=resid.rename(columns={0:'resid'})
resid['resid_sq']=resid['resid']**2
temp=pd.concat([resid,rxn_dset], axis =1)
      
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

loc_condition_summary.to_csv(f'{dir_base}/{today}_rxntime_mlm_loc_condition.csv',sep=',', index=False)


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
     
#get emm for condition  by LOC 
loc_emm=pd.concat([predictions,loc_prediction_summary], axis =1)
loc_emm_LOC_interaction=loc_emm.groupby(by=["tx_condition","LOC_YN"]).mean() #EMM for LOC by condition comparisons
loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['comparison', 'TOT_FAT', 'age','height', 'Nonwhite'])
loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
loc_emm_LOC_interaction=loc_emm_LOC_interaction.transpose()
loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(columns={0:'emm_control_noloc',1:'emm_control_loc',2:'emm_active_noloc',3:'emm_active_loc'})
loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(labels=['tx_condition','LOC_YN'],axis=0)
loc_emm_LOC_interaction=loc_emm_LOC_interaction.rename(index={"predict_value":'C(tx_condition)[T.1]:C(LOC_YN)[T.1]'})

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
 

interaction_data={'comparison':['CnoL_CL','CnoL_AnoL','CnoL_AL','CL_AnoL','CL_AL','AnoL_AL'],
                  'mean_diff':[mean_diff_CnoL_CL,mean_diff_CnoL_AnoL,mean_diff_CnoL_A,mean_diff_CL_AnoL,mean_diff_CL_AL,mean_diff_AnoL_AL],
                  "SD":[SD_CnoL_CL,SD_CnoL_AnoL,SD_CnoL_AL,SD_CL_AnoL,SD_CL_AL,SD_AnoL_AL],
                  "SE_pooled": [SE_CnoL_CL,SE_CnoL_AnoL,SE_CnoL_AL,SE_CL_AnoL,SE_CL_AL,SE_AnoL_AL]}
interaction_dframe=pd.DataFrame.from_dict(interaction_data)
interaction_dframe['cohens_d']=interaction_dframe['mean_diff']/interaction_dframe['SD']
interaction_dframe['95CI']=1.96*interaction_dframe['SE_pooled'] 
interaction_dframe['95CI_low_traditional']= interaction_dframe['cohens_d']-interaction_dframe['95CI']
interaction_dframe['95CI_up_traditional']=interaction_dframe['cohens_d']+interaction_dframe['95CI'] 

#print(loc_condition_summary)
interaction_dframe.to_csv(f'{dir_base}/{today}_rxntime_mlm_loc.csv',sep=',', index=False)

############################
#Interaction estimates. 
###########################
#save model params for interaction 
loc_summary = pd.concat([loc_param, loc_bse,loc_ci],axis=1)
loc_summary['p']=loc_model.pvalues
loc_summary=loc_summary.drop(labels=['C(tx_condition)[T.1]','C(LOC_YN)[T.1]','Intercept','C(Nonwhite)[T.1]','C(comparison)[T.2]','C(comparison)[T.3]','TOT_FAT','age','height','Group Var'],axis=0)
loc_summary=loc_summary.reset_index()
loc_summary=loc_summary.drop(columns=['index'])  
loc_emm_LOC_interaction=loc_emm_LOC_interaction.reset_index()
loc_emm_LOC_interaction=loc_emm_LOC_interaction.drop(columns=['index'])
loc_summary=pd.concat([loc_summary,loc_emm_LOC_interaction], axis =1)
loc_summary.to_csv(f'{dir_base}/{today}_rxntime_mlm_LOCinteraction.csv',sep=',', index=False)

#%%
###########################
#Compute 95% CI for EMM for each condition x LOC group. 
#entire sample, 4 group pooled SD. 
SD_pool_numerator= (((n_CnoL-1)*SD_CnoL)+((n_AnoL-1)*SD_AnoL)+((n_CL-1)*SD_CL)+((n_AL-1)*SD_AL))
SD_pool_denominator=(n_CnoL+n_AnoL+n_CL+n_AL-4)
SD_pooled=np.sqrt(SD_pool_numerator/SD_pool_denominator)    

loc_emm_LOC_interaction['95CI_low_emm_CnoL']=loc_emm_LOC_interaction['emm_control_noloc']-(1.96*(SD_pooled/(np.sqrt(n_CnoL))))
loc_emm_LOC_interaction['95CI_up_emm_CnoL']=loc_emm_LOC_interaction['emm_control_noloc']+(1.96*(SD_pooled/(np.sqrt(n_CnoL))))

loc_emm_LOC_interaction['95CI_low_emm_CL']= loc_emm_LOC_interaction['emm_control_loc']-(1.96*(SD_pooled/(np.sqrt(n_CL))))
loc_emm_LOC_interaction['95CI_up_emm_CL']=loc_emm_LOC_interaction['emm_control_loc']+(1.96*(SD_pooled/(np.sqrt(n_CL))))

loc_emm_LOC_interaction['95CI_low_emm_AnoL']= loc_emm_LOC_interaction['emm_active_noloc']-(1.96*(SD_pooled/(np.sqrt(n_AnoL))))
loc_emm_LOC_interaction['95CI_up_emm_AnoL']=loc_emm_LOC_interaction['emm_active_noloc']+(1.96*(SD_pooled/(np.sqrt(n_AnoL))))

loc_emm_LOC_interaction['95CI_low_emm_AL']= loc_emm_LOC_interaction['emm_active_loc']-(1.96*(SD_pooled/(np.sqrt(n_AL))))
loc_emm_LOC_interaction['95CI_up_emm_AL']=loc_emm_LOC_interaction['emm_active_loc']+(1.96*(SD_pooled/(np.sqrt(n_AL))))
  
loc_emm_LOC_interaction.to_csv(f'{dir_base}/{today}_rxntime_mlm_interaction_emm_CI.csv',sep=',', index=False)






