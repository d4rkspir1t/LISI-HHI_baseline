import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from scipy import linalg


def get_30joints(motion_data):
    remove_2dims= 2
    LFHD   = motion_data[:,:,2-remove_2dims:4+1-remove_2dims]
    RFHD   = motion_data[:,:,5-remove_2dims:7+1-remove_2dims]
    LBHD   = motion_data[:,:,8-remove_2dims:10+1-remove_2dims]
    RBHD   = motion_data[:,:,11-remove_2dims:13+1-remove_2dims]
    C7   = motion_data[:,:,14-remove_2dims:16+1-remove_2dims]
    T10   = motion_data[:,:,17-remove_2dims:19+1-remove_2dims]
    CLAV   = motion_data[:,:,20-remove_2dims:22+1-remove_2dims]
    STRN   = motion_data[:,:,23-remove_2dims:25+1-remove_2dims]
    LSHO   = motion_data[:,:,29-remove_2dims:31+1-remove_2dims]
    LELB   = motion_data[:,:,35-remove_2dims:37+1-remove_2dims]
    LWRA   = motion_data[:,:,41-remove_2dims:43+1-remove_2dims]
    LWRB   = motion_data[:,:,44-remove_2dims:46+1-remove_2dims]
    LFIN   = motion_data[:,:,47-remove_2dims:49+1-remove_2dims]
    RSHO   = motion_data[:,:,50-remove_2dims:52+1-remove_2dims]
    RELB   = motion_data[:,:,56-remove_2dims:58+1-remove_2dims]
    RWRA   = motion_data[:,:,62-remove_2dims:64+1-remove_2dims]
    RWRB   = motion_data[:,:,65-remove_2dims:67+1-remove_2dims]
    RFIN   = motion_data[:,:,68-remove_2dims:70+1-remove_2dims]
    LASI   = motion_data[:,:,71-remove_2dims:73+1-remove_2dims] # LFWT
    RASI   = motion_data[:,:,74-remove_2dims:76+1-remove_2dims] # RFWT
    LPSI   = motion_data[:,:,77-remove_2dims:79+1-remove_2dims] # LBWT
    RPSI   = motion_data[:,:,80-remove_2dims:82+1-remove_2dims] # RBWT
    LKNE   = motion_data[:,:,86-remove_2dims:88+1-remove_2dims]    
    LANK   = motion_data[:,:,92-remove_2dims:94+1-remove_2dims]
    LHEE   = motion_data[:,:,95-remove_2dims:97+1-remove_2dims]
    LTOE   = motion_data[:,:,98-remove_2dims:100+1-remove_2dims]
    RKNE   = motion_data[:,:,104-remove_2dims:106+1-remove_2dims]
    RANK   = motion_data[:,:,110-remove_2dims:112+1-remove_2dims]
    RHEE   = motion_data[:,:,113-remove_2dims:115+1-remove_2dims]
    RTOE   = motion_data[:,:,116-remove_2dims:118+1-remove_2dims]

    important_joints = np.concatenate((C7, CLAV, T10, STRN, 
                                 LSHO, LELB, LFIN, LWRA, LWRB,
                                 RSHO, RELB, RFIN, RWRA, RWRB,
                                 LASI, LPSI, LKNE, LANK, LHEE, LTOE,
                                 RASI, RPSI, RKNE, RANK, RHEE, RTOE,
                                 LBHD, LFHD, RBHD, RFHD), axis=2)
    return important_joints

"""Compute APE, Acceleration, and Jerk

"""

# (n_steps, n_features)
def compute_jerks(data, dim=3):
    """Compute jerk between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   jerks of each joint averaged over all frames
    """

    # Third derivative of position is jerk
    jerks = np.diff(data, n=3, axis=0)
    num_jerks = jerks.shape[0]
    num_joints = jerks.shape[1] // dim

    jerk_norms = np.zeros((num_jerks, num_joints))

    for i in range(num_jerks):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            jerk_norms[i, j] = np.linalg.norm(jerks[i, x1:x2])

    average = np.mean(jerk_norms, axis=0)

    # Take into account that frame rate 15 fps
    scaled_av = average * 15 * 15 * 15

    return scaled_av

def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   accelerations of each joint averaged over all frames
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    average = np.mean(acc_norms, axis=0)

    # Take into account that frame rate was 20 fps
    scaled_av = average * 15 * 15

    return scaled_av

# (n_steps, n_features)
def compute_APE(original, predicted, dim=3):
    """Compute Average Position Error (APE)

      Args:
          original:     array containing joint positions of original gesture
          predicted:    array containing joint positions of predicted gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   APE between original and predicted for each joint
    """

    num_frames = predicted.shape[0]
    num_joints = predicted.shape[1] // dim

    diffs = np.zeros((num_frames, num_joints))

    for i in range(num_frames):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            diffs[i, j] = np.linalg.norm(
                original[i, x1:x2] - predicted[i, x1:x2])

    return np.mean(diffs, axis=0)

def APE_Acc_JERK_GT(t_GT):
    individual_acc_GT = []
    individual_jerk_GT = []
    for GT_m in t_GT:
        
        # Acc - Ground Truth
        acc_GT_join = compute_acceleration(GT_m, dim=3)
        acc_GT_motion = np.mean(acc_GT_join, axis=0)
        individual_acc_GT.append(acc_GT_motion)

        # Jerk - Ground Truth
        jerk_GT_join = compute_jerks(GT_m, dim=1)
        jerk_GT_motion = np.mean(jerk_GT_join, axis=0)
        individual_jerk_GT.append(jerk_GT_motion)
        
    return np.mean(individual_acc_GT), np.mean(individual_jerk_GT)

def APE_Acc_JERK(t_GT, t_Gen_FC1):
    individual_APE = []
    individual_acc_GT = []
    individual_acc_predicted = []
    individual_jerk_GT = []
    individual_jerk_predicted = []
    for GT_m, predicted_m in zip(t_GT, t_Gen_FC1):
        
        # APE
        ape_joint = compute_APE(GT_m, predicted_m, dim=3)
        ape_motion = np.mean(ape_joint, axis=0)
        individual_APE.append(ape_motion)

        # Acc - Ground Truth
        acc_GT_join = compute_acceleration(GT_m, dim=3)
        acc_GT_motion = np.mean(acc_GT_join, axis=0)
        individual_acc_GT.append(acc_GT_motion)

        # Acc - Generated 
        acc_gen_join = compute_acceleration(predicted_m, dim=3)
        acc_gen_motion = np.mean(acc_gen_join, axis=0)
        individual_acc_predicted.append(acc_gen_motion)
              
        # Jerk - Ground Truth
        jerk_GT_join = compute_jerks(GT_m, dim=1)
        jerk_GT_motion = np.mean(jerk_GT_join, axis=0)
        individual_jerk_GT.append(jerk_GT_motion)

        # Jerk - Generated 
        jerk_gen_join = compute_jerks(predicted_m, dim=1)
        jerk_gen_motion = np.mean(jerk_gen_join, axis=0)
        individual_jerk_predicted.append(jerk_gen_motion)
        
    return np.mean(individual_APE), np.mean(individual_acc_predicted), np.mean(individual_jerk_predicted), np.mean(individual_acc_GT), np.mean(individual_jerk_GT)


def get_scores(generated_feats, real_feats):

    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    ####################################################################
    # frechet distance
    frechet_dist = frechet_distance(generated_feats, real_feats)

    ####################################################################
    # # distance between real and generated samples on the latent feature space1    dists = []
    # for i in range(real_feats.shape[0]):
    #     d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
    #     dists.append(d)
    # feat_dist = np.mean(dists)

    # return frechet_dist, feat_dist
    return frechet_dist
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)