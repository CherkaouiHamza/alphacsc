import os
from hcp_builder.dataset import (get_data_dirs, download_experiment,
                                 fetch_subject_list)


EPOCH_DUR_HCP = 12.0
DUR_RUN_HCP = 3*60 + 34
N_SCANS_HCP = 284
TR_HCP = DUR_RUN_HCP / float(N_SCANS_HCP)


def get_hcp_fmri_fname(subject_id, anat_data=False):
    """Return the tfMRI filename."""
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    if not os.path.exists(path):
        download_experiment(subject=subject_id, data_dir=None,
                            data_type='task', tasks='MOTOR', sessions=None,
                            overwrite=True, mock=False, verbose=10)
    fmri_path = path
    fmri_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL.nii.gz']
    for dir_ in fmri_dirs:
        fmri_path = os.path.join(fmri_path, dir_)
    anat_path = path
    anat_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL_SBRef.nii.gz']
    for dir_ in anat_dirs:
        anat_path = os.path.join(anat_path, dir_)
    if anat_data:
        return fmri_path, anat_path
    else:
        return fmri_path
