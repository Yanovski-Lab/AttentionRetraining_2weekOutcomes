#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:57:01 2023

@author: MegNParker
"""

import mne
import pandas as pd 
import glob
import os, os.path as op
import numpy as np
import copy

#%%

subjid="..."
session=1
sessions=[1,2]
surf_flag=True #FALSE
bids_dir="..."
subjects_dir=op.join(bids_dir,"derivatives/freesurfer/subjects")
subjids=glob.glob(op.join(bids_dir,"sub-*"))
subjids=[op.basename(i) for i in subjids]
out_dir=f"{bids_dir}/derivatives/mne-bids-pipeline/{subjid}/ses-{session}/meg"
l_freq=13
h_freq=35

if 'n_jobs' in os.environ:
    n_jobs = os.environ['n_jobs']
else:
    n_jobs = 4


#%% initialize logger

log_dir="{bids_dir}/derivatives/logs"
if not os.path.exists(log_dir): os.mkdir(log_dir)
import logging 
logger=logging.getLogger()

def get_subj_logger(subjid, session, log_dir=None):
     '''Return the subject specific logger.
     This is particularly useful in the multiprocessing where logging is not
     necessarily in order'''
     fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
     sub_ses = f'{subjid}_ses_{session}'
     subj_logger = logging.getLogger(sub_ses)
     if subj_logger.handlers != []: # if not first time requested, use the file handler already defined
         tmp_ = [type(i) for i in subj_logger.handlers ]
         if logging.FileHandler in tmp_:
             return subj_logger
     else: # first time requested, add the file handler
         fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_ses-{session}_log.txt')
         fileHandle.setLevel(logging.INFO) #level issue?
         fileHandle.setFormatter(logging.Formatter(fmt)) 
         subj_logger.addHandler(fileHandle)
         subj_logger.info('Initializing subject log')
     return subj_logger 
 
#%% Identify relevant events and create epochs 

####find folders that contain AMREA task sourcedata. 
megdir='...'
os.chdir(megdir)
dsets=glob.glob('????????/*.ds')
 
def get_subjid(fname):
    tmp = op.basename(fname)
    return tmp.split('_')[0]

def get_date(fname):
    tmp = op.dirname(fname)
    return tmp

#create list of subjects and dates. 
dframe=pd.DataFrame(dsets, columns=['fname'])
dframe['subjid']=dframe.fname.apply(get_subjid)
dframe['date']=dframe.fname.apply(get_date)
dframe=dframe.loc[dframe['fname'].str.contains("AREMA")]

#get first session date. 
dated_dframe= dframe.groupby('subjid')['date'].min().apply(''.join).reset_index()
dated_dframe= dated_dframe.rename({'date':'session1'},axis=1)

#add first session date to the larger dataset.
dframe_session=dframe.merge(dated_dframe, on='subjid', validate="many_to_one")

#determine session #. 
dframe_session['date']= dframe_session['date'].astype(int)
dframe_session['session1']=  dframe_session['session1'].astype(int)
dframe_session['days_from_sess1']= (dframe_session['date'] - dframe_session['session1']).astype(int)

dated_dframe= dframe_session.drop_duplicates(subset=['subjid','date'])
dated_dframe['session']=dated_dframe.groupby('subjid')['days_from_sess1'].rank(method='first',ascending=True)
dated_dframe['session']=dated_dframe['session'].astype(int)

dated_dframe=dated_dframe.drop(labels=['session1','days_from_sess1','fname'],axis=1)
dframe_session=dframe_session.merge(dated_dframe, on=['subjid', 'date'], validate="many_to_one")
  

#create function that creates epochs. 
def get_epochs(subjid,f_name,session, meg_fname):
              
        raw = mne.io.read_raw_ctf(meg_fname, clean_names=False, system_clock='ignore')
        events, event_ids = mne.events_from_annotations(raw) 
        epochs = mne.Epochs(raw, events, event_id=dict(HFvLF_hfb=5, HFvLF_nohfb=6,HFvNF_hfb=7,HFvNF_nohfb=8,LFvNF_lfb=13,LFvNF_nolfb=14), tmin=-.1, tmax=5, preload=True, baseline=(-.1, 0))
        epochs_fname=f"{out_dir}/sub-{subjid}_ses-{session}_task-AREMA_all-epo.fif"
        epochs.save(epochs_fname,overwrite=True)
        print (subjid,session,"done")

#loop over sujects and sessions to create epochs.
for i,row in dframe_session.iterrows():    
    subjid=row['subjid']
    f_name=row['fname']
    session=row['session']
    meg_fname=f'{bids_dir}/sourcedata/{f_name}'
    logger=get_subj_logger(subjid, session, log_dir=log_dir)
    try:
        get_epochs (subjid,f_name,session, meg_fname)
    except BaseException as e:
        logger.exception(f"epochs not corrected::{e}")
    
#%%   Create separate cortical and subcortical forward models 

def create_vol_fwd(subjid,session):

        fname_model=f'{subjects_dir}/{subjid}/bem/{subjid}-5120-bem-sol.fif'
        fname_aseg=op.join(subjects_dir,subjid,"mri/aseg.mgz")
        
        vol_src = mne.setup_volume_source_space(
            subject=subjid,
            mri=fname_aseg,
            pos=5.0,
            bem=fname_model,
            subjects_dir=subjects_dir,
            add_interpolator=True,
            verbose=True)
        

        fname_evoked=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_ave.fif"
        fname_trans=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_trans.fif'
        fwd_vol=mne.make_forward_solution(fname_evoked,fname_trans,vol_src,fname_model,mindist=5.0,  meg=True,eeg=False,n_jobs=4)
        
        fname_fwd_vol=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_vol.fif"
        mne.write_forward_solution(fname=fname_fwd_vol, fwd=fwd_vol,overwrite=True)
        
        return fwd_vol
    
def create_surf_fwd(subjid,session):    

    fname_model=f'{subjects_dir}/{subjid}/bem/{subjid}-5120-bem-sol.fif'

    src_fname=f'{subjects_dir}/{subjid}/bem/{subjid}-oct6-src.fif'
    surf_src=mne.read_source_spaces(src_fname)
    
    fname_evoked=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_ave.fif"
    fname_trans=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_trans.fif'
    fwd_surf=mne.make_forward_solution(fname_evoked,fname_trans,surf_src,fname_model,mindist=5.0,  meg=True,eeg=False,n_jobs=4)
    
    fname_fwd_surf=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_surf.fif"
    mne.write_forward_solution(fname=fname_fwd_surf, fwd=fwd_surf,overwrite=True)
    
    return fwd_surf

     
#%% create beamformer filters

def return_beamformer_filter(epochs=None,
                  bem=None,
                  fwd=None,
                  regularization=0.05,
                  return_stc_epochs=False,
                  surf_flag=False):
    
    if surf_flag==True:
        fname_fwd=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_surf.fif"
        fname_filter=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_filter_surf_lcmv_beta.h5'
    else:
        fname_fwd=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_vol.fif"
        fname_filter=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_filter_vol_lcmv_beta.h5'
    fwd=mne.read_forward_solution(fname_fwd)
    
   
    epochs=mne.read_epochs(f"{out_dir}/{subjid}_ses-{session}_task-AREMA_all-epo.fif")
    beta_epochs=epochs.filter(l_freq=l_freq, h_freq=h_freq)

        
    #use same cov for both volume and surface models. 
    noise_cov = mne.compute_covariance(epochs=beta_epochs, method='shrunk', cv=5, n_jobs=n_jobs,
                                       tmax=0) #pre stim
    data_cov = mne.compute_covariance(epochs=beta_epochs, method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=-.1) #runs for the whole trial
    filters = make_lcmv(beta_epochs.info, fwd, data_cov=data_cov, noise_cov=noise_cov, rank=None, reg=0.05, 
                        pick_ori='max-power', weight_norm='unit-noise-gain')
    
    filters.save(fname_filter, overwrite=True)
    return filters

#%% project data through beamformer filters for each time component and trial type

def create_stc(subjid,session, surf_flag):
    
    if surf_flag==True:
        fname_filter=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_filter_surf_lcmv_beta.h5'
        model="surf"
    else:
        fname_filter=f'{out_dir}/{subjid}_ses-{session}_task-AREMA_filter_vol_lcmv_beta.h5'
        model="vol"
    filters=mne.beamformer.read_beamformer(fname_filter)

    epochs=mne.read_epochs(f"{out_dir}/{subjid}_ses-{session}_task-AREMA_all-epo.fif")
    beta_epochs=epochs.filter(l_freq=l_freq, h_freq=h_freq)

     
    
    #trial type = high palatable food-non food. 
    ucr_HFvNF_hfb_cov=mne.compute_covariance(beta_epochs['HFvNF_hfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_HFvNF_hfb_cov=mne.compute_covariance(beta_epochs['HFvNF_hfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    ucr_HFvNF_nohfb_cov=mne.compute_covariance(beta_epochs['HFvNF_nohfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_HFvNF_nohfb_cov=mne.compute_covariance(beta_epochs['HFvNF_nohfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    
    stc_HFvNF_ucr_hfb=apply_lcmv_cov(ucr_HFvNF_hfb_cov,filters)
    stc_HFvNF_ucr_nohfb=apply_lcmv_cov(ucr_HFvNF_nohfb_cov,filters)
    stc_HFvNF_ucr_diff = copy.deepcopy(stc_HFvNF_ucr_hfb)
    stc_HFvNF_ucr_diff._data=np.log(stc_HFvNF_ucr_hfb._data)-np.log(stc_HFvNF_ucr_nohfb._data)
    
    stc_HFvNF_ad_hfb=apply_lcmv_cov(ad_HFvNF_hfb_cov,filters)
    stc_HFvNF_ad_nohfb=apply_lcmv_cov(ad_HFvNF_nohfb_cov,filters)
    stc_HFvNF_ad_diff = copy.deepcopy(stc_HFvNF_ad_hfb)
    stc_HFvNF_ad_diff._data = np.log(stc_HFvNF_ad_hfb._data) - np.log(stc_HFvNF_ad_nohfb._data)
    
    tag='HFvNF_ucr_diff'    
    stc_HFvNF_ucr_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)
    
    tag='HFvNF_ad_diff'    
    stc_HFvNF_ad_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)
    
    #trial type = high palatable food - low palatable food,.
    ucr_HFvLF_hfb_cov=mne.compute_covariance(beta_epochs['HFvLF_hfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_HFvLF_hfb_cov=mne.compute_covariance(beta_epochs['HFvLF_hfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    ucr_HFvLF_nohfb_cov=mne.compute_covariance(beta_epochs['HFvLF_nohfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_HFvLF_nohfb_cov=mne.compute_covariance(beta_epochs['HFvLF_nohfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    
    stc_HFvLF_ucr_hfb=apply_lcmv_cov(ucr_HFvLF_hfb_cov,filters)
    stc_HFvLF_ucr_nohfb=apply_lcmv_cov(ucr_HFvLF_nohfb_cov,filters)
    stc_HFvLF_ucr_diff = copy.deepcopy(stc_HFvLF_ucr_hfb)
    stc_HFvLF_ucr_diff._data=np.log(stc_HFvLF_ucr_hfb._data)-np.log(stc_HFvLF_ucr_nohfb._data)
    
    stc_HFvLF_ad_hfb=apply_lcmv_cov(ad_HFvLF_hfb_cov,filters)
    stc_HFvLF_ad_nohfb=apply_lcmv_cov(ad_HFvLF_nohfb_cov,filters)
    stc_HFvLF_ad_diff = copy.deepcopy(stc_HFvLF_ad_hfb)
    stc_HFvLF_ad_diff._data=np.log(stc_HFvLF_ad_hfb._data)-np.log(stc_HFvLF_ad_nohfb._data)
    
    tag='HFvLF_ucr_diff'    
    stc_HFvLF_ucr_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)
    
    tag='HFvLF_ad_diff'    
    stc_HFvLF_ad_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)
    
    #trial type = low palatable food - non food. 
    ucr_LFvNF_lfb_cov=mne.compute_covariance(beta_epochs['LFvNF_lfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_LFvNF_lfb_cov=mne.compute_covariance(beta_epochs['LFvNF_lfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    ucr_LFvNF_nolfb_cov=mne.compute_covariance(beta_epochs['LFvNF_nolfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0, tmax=.250) #runs for the whole trial
    ad_LFvNF_nolfb_cov=mne.compute_covariance(beta_epochs['LFvNF_nolfb'], method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=.25, tmax=.5)
    
    stc_LFvNF_ucr_lfb=apply_lcmv_cov(ucr_LFvNF_lfb_cov,filters)
    stc_LFvNF_ucr_nolfb=apply_lcmv_cov(ucr_LFvNF_nolfb_cov,filters)
    stc_LFvNF_ucr_diff = copy.deepcopy(stc_LFvNF_ucr_lfb)
    stc_LFvNF_ucr_diff._data=np.log(stc_LFvNF_ucr_lfb._data)-np.log(stc_LFvNF_ucr_nolfb._data)
    
    stc_LFvNF_ad_lfb=apply_lcmv_cov(ad_LFvNF_lfb_cov,filters)
    stc_LFvNF_ad_nolfb=apply_lcmv_cov(ad_LFvNF_nolfb_cov,filters)
    stc_LFvNF_ad_diff = copy.deepcopy(stc_LFvNF_ad_lfb)
    stc_LFvNF_ad_diff._data=np.log(stc_LFvNF_ad_lfb._data)-np.log(stc_LFvNF_ad_nolfb._data)
    
    tag='LFvNF_ucr_diff'    
    stc_LFvNF_ucr_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)
    
    tag='LFvNF_ad_diff'    
    stc_LFvNF_ad_diff.save(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_{model}_beta',overwrite=True)


#%%  parcel processing

def extract_surf_roi_matrix(subjid, session):  
    sub_labels=mne.read_labels_from_annot(subjid, subjects_dir=subjects_dir,parc='aparc') #read in the labels.

    #Comprehensive list of surface ROIs want to extract. 
    regions= ['caudalanteriorcingulate','caudalmiddlefrontal','frontalpole',
              'lateralorbitofrontal','medialorbitofrontal','rostralanteriorcingulate',
              'rostralmiddlefrontal','superiorfrontal','parsopercularis','parsorbitalis',
              'parstriangularis']

    
    ROIs= [i for i in sub_labels if i.name[:-3] in regions]
    
    fname_fwd_surf=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_surf.fif"
    fwd=mne.read_forward_solution(fname_fwd_surf)
    src=fwd["src"]
    
    dframe=pd.DataFrame(index=ROIs)
    dframe_surf=pd.DataFrame(columns=['condition','ts'])

    tag='HFvNF_ucr_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc, ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    tag='HFvNF_ad_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc,  ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    tag='HFvLF_ucr_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc, ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    tag='HFvLF_ad_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc,  ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    tag='LFvNF_ucr_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc, ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    tag='LFvNF_ad_diff'    
    stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_surf_beta')
    dframe["ts"]=mne.extract_label_time_course(stc,  ROIs, src, mode="mean")
    dframe['condition']=tag
    dframe_surf=pd.concat([dframe_surf,dframe],join='outer')
    dframe=pd.DataFrame(index=ROIs)

    dframe_surf['subjid']=subjid
    return dframe_surf


def extract_vol_roi_matrix(subjid, session):     
     
    #Comprehensive list of subcortical ROIs want to extract. 
      vol_regions = (subjects_dir+"/"+subjid+'/mri/aseg.mgz', 
             ['Left-Caudate','Left-Putamen','Left-Pallidum',
               'Right-Caudate','Right-Putamen','Right-Pallidum'])
    
    
      fname_fwd_vol=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_fwd_vol.fif"
      fwd=mne.read_forward_solution(fname_fwd_vol)
      src=fwd["src"]
    
      dframe=[]
      dframe=pd.DataFrame(index=vol_regions[1])
      
      dframe_vol=pd.DataFrame(columns=['condition','ts'])
      
      tag='HFvNF_ucr_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])
      
      tag='HFvNF_ad_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])

      tag='HFvLF_ucr_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])

      tag='HFvLF_ad_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])

      tag='LFvNF_ucr_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])

      tag='LFvNF_ad_diff'    
      stc=mne.read_source_estimate(f'{out_dir}/{subjid}_ses-{session}_task-AREMA_lcmv_{tag}_vol_beta-vl.stc')
      dframe["ts"]=mne.extract_label_time_course(stc,  vol_regions, src, mode="mean") 
      dframe['condition']=tag
      dframe_vol=pd.concat([dframe_vol,dframe],join='outer')
      dframe=pd.DataFrame(index=vol_regions[1])

      dframe_vol['subjid']=subjid
      return dframe_vol


#%% loop over subjects and sessions to create indvidual stc dframes 

for subjid in subjids:
    for session in sessions: 
        logger=get_subj_logger(subjid, session, log_dir=log_dir)
        try:
            create_vol_fwd(subjid,session)
            create_surf_fwd(subjid,session)
        except BaseException as e:
            logger.exception(f"create fwd models::{e}")
        try:
            filter_vol=return_beamformer_filter(surf_flag=False)
            filter_surf=return_beamformer_filter(surf_flag=True)
        except BaseException as e:
            logger.exception(f"create beamformer filters::{e}")
        try:
            create_stc(subjid,session, surf_flag=False)
            create_stc(subjid,session, surf_flag=True)
        except BaseException as e:
            logger.exception(f"no stcs::{e}")
        try:
            dframe_surf=extract_surf_roi_matrix(subjid, session)  
            dframe_vol=extract_vol_roi_matrix(subjid, session)
            dframe_final=pd.concat([dframe_surf,dframe_vol])
            out_fname=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_mixed_lcmv_output_beta.csv" 
            dframe_final.to_csv(out_fname)
        except BaseException as e:
            logger.exception(f"extract ROIs::{e}")
         
            
#%% combine sct files for group level statistics.

# Import date class from datetime module
from datetime import date
today = date.today()

#create output files
outname_ts=f"/.../masterdata_task-AREMA_mixed_lcmv_output_{today}_beta.csv" 
out_ts=pd.DataFrame(columns=['Unnamed: 0', 'condition', 'ts', 'subjid', 'session'])
out_ts.to_csv(outname_ts, index=False)
 
for subjid in subjids:
    for session in sessions:         
        subj_csv=f"{out_dir}/{subjid}_ses-{session}_task-AREMA_mixed_lcmv_output_beta.csv" 
        if os.path.isfile(subj_csv): 
            data = pd.read_csv(subj_csv)
            data['session']=session
            df=pd.read_csv(outname_ts)
            df = pd.concat([df, data])
            df.to_csv(outname_ts, index=False)
            df=open('stc_output','a')
            df.write(f'\n{subjid},{session}, stc success')
            df.close()
        else:
            df=open('stc_output','a')
            df.write(f'\n{subjid},{session},stc failure')
            df.close()

          