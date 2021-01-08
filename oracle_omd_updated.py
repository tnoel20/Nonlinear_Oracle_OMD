mport matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import os
import math
from pyod.models.loda import LODA
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from oc_data_load import CIFAR10_Data
#from vanilla_ae import get_vanilla_ae
from classifier import get_resnet_18_classifier


def load_data(split=0, normalize=False):
    ''' 
    Known/Unknown split semantics can be found in download_cifar10.py
    in the following git repo:
    https://github.com/lwneal/counterfactual-open-set

    I use a modified download script to produce csv files instead of
    JSON. Refer to download_cifar10_to_csv.py for further details

    NOTE: There are 5 possible splits that can be used here,
          the default split number is 0 unless specified otherwise.
    '''
    if normalize:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247 , 0.243 , 0.261))
        ])
    else:
        # Just this by default
        transform = T.ToTensor()
        
    kn_train   = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}a.dataset'.format(split),
                              fold='train',
                              transform=transform)
    kn_val    = CIFAR10_Data(root_dir='data/cifar10',
                             csv_file='data/cifar10-split{}a.dataset'.format(split),
                             fold='val',
                             transform=transform)
    kn_test    = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}a.dataset'.format(split),
                              fold='test',
                              transform=transform)
    
    unkn_train = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}b.dataset'.format(split),
                              fold='train',
                              transform=transform)
    unkn_val  = CIFAR10_Data(root_dir='data/cifar10',
                             csv_file='data/cifar10-split{}b.dataset'.format(split),
                             fold='val',
                             transform=transform)
    unkn_test  = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}b.dataset'.format(split),
                              fold='test',
                              transform=transform)

    #kn_train = torch.utils.data.DataLoader(kn_train, batch_size=4, pin_memory=True)
    #kn_val = torch.utils.data.DataLoader(kn_val, batch_size=4, pin_memory=True)
    #kn_test = torch.utils.data.DataLoader(kn_test, batch_size=4, pin_memory=True)
    #unkn_train = torch.utils.data.DataLoader(unkn_train, batch_size=4, pin_memory=True)
    #unkn_val = torch.utils.data.DataLoader(unkn_val, batch_size=4, pin_memory=True)
    #unkn_test = torch.utils.data.DataLoader(unkn_test, batch_size=4, pin_memory=True)

    return (kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test)


def omd(train_latent_data, latent_data, anom_classes, learning_rate=1e-2):
    '''
    Training a linear anomaly detector

    Originally developed to train a linear anomaly
    detector on latent representations of labeled images

    Parameters
    ----------
    data: data instances packed into a dataframe

    anom_classes: a list of classes corresponding
                  to anomaly classification (should
                  be a list of unique strings)
    '''
    N = len(latent_data)
    T = 1000#=N
    labels = latent_data['label']
    # TODO: Initialize this with LODA values
    theta, clf = get_weight_prior(train_latent_data)
    data = loda_transform(clf, latent_data)
    print('Weight Prior: {}'.format(theta))
    # Note that this is usually implemented as an
    # online algorithm so number of time steps (steps
    # of this outer loop) are usually ambiguous. Here
    # we have a dataset of some fixed size, so we
    # will start by running a single epoch over the
    # dataset
    # TODO: Mess around with reducing this number
    for i in range(T):
        w = get_nearest_w(theta)
        d_anom, idx = get_max_anomaly_score(w, data)
        y = get_feedback(labels.iloc[idx], anom_classes)
        # TAG
        #data.drop(data.index[idx])
        np.delete(data, idx)
        # linear loss function
        # loss = -y*np.dot(w,d_anom)
        theta = theta + learning_rate*y*d_anom#- learning_rate*y*d_anom

    return w, clf


def loda_transform(loda_clf, data_df):
    '''
    For each training example, for each histogram, project training
    example into appropriate histogram bin. Mark this bin as 1 and
    the rest as zero and then proceed to multiply the resulting
    vector by the negative log probability associated with the
    histogram bin. Finally ravel the constructed modified
    histogram matrix; this yields one processed training example.
    '''
    N = len(data_df) 
    X = data_df.drop(columns=['label'])
    hists = loda_clf.histograms_
    num_hists = len(hists)
    data = X.to_numpy()
    #transformed_data = np.zeros(N)
    transformed_data = []
    for i in range(N):
        # Create a copy that we can modify to yield
        # the ith training example
        ith_hists = np.copy(hists)
        for j in range(num_hists):
            wj = 0
            projected_data = loda_clf.projections_[j,:].dot(data[i])
            # Assumes that this also works for finding a single index
            ind = np.searchsorted(loda_clf.limits_[j, :loda_clf.n_bins - 1],
                                  projected_data, side='left')
            # This is currently zero...
            #print(ith_hists[j,ind])
            ##if ith_hists[j,ind] > 0:
                #print("Updating wj")
                ##wj = -math.log2(ith_hists[j,ind])
            ith_hists[j,ind] = 1
            #zero_inds = np.where(ith_hists[j] != 1)
            #print(ith_hists[j])
            #ith_hists[j, np.arange(len(ith_hists[i]))!=ind] = 0 
            ith_hists[j, np.arange(len(ith_hists[j]))!=ind] = 0
            #print(wj)
            ##ith_hists[j] *= wj

        #tranformed_data[i] = np.ravel(ith_hists)
        transformed_data.append(np.ravel(ith_hists))         

    #print(transformed_data)
    return np.array(transformed_data)


def get_feedback(label, anom_classes):
    '''Checks for membership of label
    in given set of anomaly classes.
    If the instance is anomalous,
    returns 1 else returns -1'''
    y = -1
    if label in anom_classes:
        y = 1
        
    return y

        
def get_max_anomaly_score(w, D_X):
    '''Returns the element in the dataset
    with the largest anomaly score    '''
    N = len(D_X)
    #D_X = D.drop(columns=['label']).to_numpy()
    x_curr = np.dot(w, D_X[0])#-w, D_X[0])#D_X.iloc[0])
    x_max = x_curr
    idx = 0
    for i in range(N):
        x_curr = np.dot(w, D_X[i])#-w, D_X[i])#D_X.iloc[i])
        print('x_curr: {}'.format(x_curr))
        if x_curr > x_max:
            print('x_curr is new x_max!------------------------------------------------------')
            x_max = x_curr
            idx = i

    return x_max, idx


def get_nearest_w(theta):
    '''Returns the weight vector in the space
    of d-dimensional vectors with positive real 
    numbers that is closest to the given theta'''
    return relu(theta)


def relu(x_vec):
    '''Just makes negative elements 0'''
    x = np.copy(x_vec)
    x[x < 0] = 0
    return x


def train_oracle_latent_rep():
    pass


def get_plain_ae(kn_train, kn_val, filename='plain_ae.pth'):
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20
    
    #device = torch.device("cuda")
    model = get_vanilla_ae(kn_train, kn_val, filename)
    return model


def get_weight_prior(X_latent):
    '''
    Get weight vector prior using LODA (Pevny16)
    
    Parameters
    ----------
    X_val_latent : numpy array
        Describes the latent representation of images in the
        validation set. Note: This contains known and unknown
        examples.

    Returns
    -------
    ndarray: Concatenated one-hot histogram vectors, where a given bin
    is a 1 if it has the greatest probability. Used as the
    prior in training a linear anomaly detector on all classes
    given latent representation from a model trained on only 6.
    object: Fitted LODA estimator.
    '''

    X = X_latent.drop(columns=['label'])
    n_bins = 10
    n_random_proj = 100
    clf = LODA(n_bins=n_bins, n_random_cuts=n_random_proj)
    model = clf.fit(X)
    hists = model.histograms_
    weight_prior = np.copy(model.histograms_)
    weight_prior = np.log2(weight_prior)
    #print('hists: {}'.format(hists))
    #print('hists shape: {}'.format(hists.shape))
    #print('weight_prior: {}'.format(weight_prior))
    #print('weight_prior shape: {}'.format(weight_prior.shape))
    # For each histogram, get max element index, and calculate
    # -log(\hat{p}), where \hat{p} is the value of the max element
    ##for i in range(n_random_proj):
        #print(weight_prior[i])
        #max_ind = np.argmax(hists[i])
        #print('max_ind: {}'.format(max_ind))
        # Calculate the weight associated with this element
        #print('histogram: {}'.format(hists[i]))
        #print('In weight prior: arg of log2: {}'.format(hists[i,max_ind]))
        ##log_probs = -math.log2(hists[i,max_ind])
        #print('wi: {}'.format(wi))
        #print('weight_prior (before step 1): {}'.format(weight_prior))
        ##weight_prior[i,max_ind] = 1
        #print('weight_prior (after step 1): {}'.format(weight_prior))
        #print('model hists: {}'.format(hists))
        ##zero_inds = np.arange(len(weight_prior[i])) != max_ind
        ##weight_prior[i, np.arange(len(weight_prior[i]))!=max_ind] = 0 
        #print('weight_prior (step 2): {}'.format(weight_prior))
        #print('model hists: {}'.format(hists))
        ##weight_prior[i] *= wi

    return np.ravel(weight_prior), model 
         

def get_features_t_stats(X_latent):
    '''
    Get weight vector prior using LODA (Pevny16)
    
    Parameters
    ----------
    X_val_latent : numpy array
        Describes the latent representation of images in the
        validation set. Note: This contains known and unknown
        examples.

    Returns
    -------
    t-stat associated with each feature. Used as the
    prior in training a linear anomaly detector on all classes
    given latent representation from a model trained on only 6.
    
    '''
    X = X_val_latent.drop(columns=['label'])
    N = len(X)
    n_features = len(X.iloc[0])
    t_vec = []
    n_bin = N // 50
    n_rand_proj = N // 2
    clf = LODA(n_bins=n_bin, n_random_cuts=n_rand_proj)
    model = clf.fit(X)
    hists = model.histograms_
    projections = model.projections_
    features_used = feature_use(projections)
    for i in range(n_features):
        Ij      = np.nonzero(features_used[:,i] == 1)[0]
        Ij_bar  = np.nonzero(features_used[:,i] == 0)[0]
        used_size   = len(Ij)
        unused_size = len(Ij_bar)
        print('Ij length: {}'.format(len(Ij)))
        print('Ij_bar length: {}'.format(len(Ij_bar)))
        mu_j, var_j = get_stats(model, i, Ij, X)
        bar_mu_j, bar_var_j = get_stats(model, i, Ij_bar, X)
        t_vec.append((mu_j-bar_mu_j)/((var_j/used_size)+(bar_var_j/unused_size)))

    print(len(t_vec) == n_features)
    return t_vec.to_numpy()


def feature_use(projections):
    num_proj = len(projections)
    num_features = len(projections[0])
    use_matrix = np.zeros((num_proj, num_features))
    for i in range(num_proj):
        for j in range(num_features):
            if projections[i][j] != 0:
                use_matrix[i][j] = 1

    return use_matrix


def get_stats(model, feat_idx, ensemble_indices, X_df):
    '''Extracts t-statistic mentioned in Pevny's paper; Mean and variance
    calculations need to be verfied. Currently averaging across all
    probability mappings for all samples for a given feature to calculate
    mean and variance of the negative log probabilities.'''
    X = X_df.to_numpy()
    projections = model.projections_
    histograms  = model.histograms_
    num_samples = len(X)
    n_projections = len(projections)
    mean = np.zeros(num_samples)#n_projections)
    var = np.zeros(num_samples)#n_projections)
    neg_log_probs = np.zeros((n_projections, num_samples))
    for i in range(n_projections):
        if i in ensemble_indices:
            projected_data = projections[i, :].dot(X.T)
            inds = np.searchsorted(model.limits_[i, :model.n_bins - 1],
                                   projected_data, side='left')
            neg_log_probs[i, inds] = -np.log(model.histograms_[i, inds])

    # Gives us mean -log\hat{p} across all ensemble bins
    # for each sample
    for i in range(num_samples):
        mean[i] = np.mean(neg_log_probs[:,i])
        var[i] = np.var(neg_log_probs[:,i])

    return np.mean(mean), np.var(var)

                     
def construct_column_labels(data_sample):
    '''Builds a list of labels that will be used to label
    columns in dataframe representing latent data'''
    num_features = len(data_sample)
    # This could be, for example, the size of the
    # latent space representation
    feature_list = []
    for i in range(num_features):
        feature_list.append('f_{}'.format(str(i)))
        
    feature_list.append('label')
    return feature_list


def concat_design_and_target(dataset): #, metadata):
    ''' 
    Embeds labels with training examples. 
    Utility for building latent representation dataset
    '''
    concat_data = []
    df = dataset.frame
    num_examples = len(dataset)
    for i in range(num_examples):
        concat_data.append([dataset[i], df.iloc[i]['label']])
         
    return concat_data


def construct_latent_set(model, kn_dataset, unkn_dataset):
    '''Build dataset from latent representation given
    a model that acts as the encoder and a dataset of
    raw data that is transformed by the encoder'''
    kn_X_y = concat_design_and_target(kn_dataset)#, metadata)
    unkn_X_y = concat_design_and_target(unkn_dataset)
    kn_unkn_X_y = torch.utils.data.ConcatDataset([kn_X_y, unkn_X_y])
    loader = torch.utils.data.DataLoader(kn_unkn_X_y, batch_size=1, shuffle=True)
    col_labels_loaded = False
    col_labels = []
    embed_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader = tqdm(loader)
    
    # NOTE: Each image batch consists of one image
    for i, (img_batch, label) in enumerate(loader):
        # If pooling was used, we get data AND indices, so we
        # need "don't care" notation as second returned var
        img_batch = img_batch.to(device)
        latent_batch = model.get_latent(img_batch)
        embedding = torch.reshape(torch.squeeze(latent_batch), (-1,))
        # TODO: Append this embedding to a pd dataframe with its label
        if col_labels_loaded == False:
            col_labels = construct_column_labels(embedding)
            val_latent_rep_df = pd.DataFrame(columns=col_labels)
            col_labels_loaded = True
            print("Populating Latent Dataframe")

        embedding = embedding.tolist()
        embedding.append(label[0])
        val_latent_rep_df.loc[i] = embedding

    print("Latent Dataframe Loading Complete")
    print(val_latent_rep_df)
    return val_latent_rep_df


def test_results(test_data, weights, y_class, anom_classes):
    '''
    Tests the linear anomaly detector on the test
    data specified.

    Parameters
    ----------
    test_data: Dataframe
        Contains test data and labels (in final column, entitled 'label')

    weights: numpy array
        Learned weights of linear anomaly detector

    anom_classes: str list
        A list of classes deemed anomalous


    Returns
    -------
    scores: numpy array
        Classifications on a per-example basis (+1: anomalous; -1: nominal)

    y: numpy array
        Actual classification of each example  ("" "")
    '''
    num_examples = len(test_data)
    X = test_data#.drop(columns=['label'])
    #y_class = test_data['label']
    # get_feedback(label, anom_classes)
    y = np.zeros(num_examples)
    scores = np.zeros(num_examples)
    for i in range(num_examples):
        y[i] = get_feedback(y_class[i], anom_classes)
    #data_iter = tqdm(X.iterrows())
    for i, example in enumerate(X):
        scores[i] = np.dot(-weights, example)#-weights, example)

    return scores, y


def plot_auroc(y_actual, scores):
    fpr, tpr, thresholds = roc_curve(y_actual, scores, pos_label=1)
    plt.plot(fpr,tpr)
    plt.show()

def main():
    CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

    SPLIT = 0
    
    # The 2nd dimension of this list contains indices of anomalous
    # classes corresponding to the split index, represented by
    # the corresponding index in the first dimension
    #
    # e.g. SPLIT = 0 means that our anomaly indices are 'cat', 'frog',
    # 'horse', and 'ship'
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]

    anom_classes = [CIFAR_CLASSES[i] for i in splits[SPLIT]]

    # Get datasets of known and unknown classes
    kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test = load_data(SPLIT)
    
    # binary or multiclass category detector??

    # WHAT IS ACTUALLY RETURNED HERE???
    #kn_ae = get_plain_ae(kn_train, kn_val,'kn_std_ae_split_{}.pth'.format(0))
    kn_classifier = get_resnet_18_classifier(kn_train, kn_val)

    ''' <><><><><><><> USE THIS IF YOU NEED AE THAT IS TRAINED ON KN/UNKN <><><><><>
    # Training plain autoencoder on all training data
    kn_unkn_train = torch.utils.data.ConcatDataset([kn_train,unkn_train])
    # This preserves metadata
    # MIGHT NOT NEED kn_unkn_val_frame = pd.concat([kn_val.frame, unkn_val.frame])
    kn_unkn_val = torch.utils.data.ConcatDataset([kn_val,  unkn_val  ])
    kn_unkn_ae  = get_plain_ae(kn_unkn_train, kn_unkn_val,
                              'kn_unkn_std_ae_split_{}.pth'.format(0))

    '''
    #if os.path.isfile('weights_oracle_feedback.txt'):
    #    # Load weights
    #    with open('weights_oracle_feedback.txt', 'rb') as f:
    #        w = np.load(f)
        
    #else:
    # Get latent set used to train linear anomaly detector from the
    # validation set comprised of all classes. Note that we are
    # using the autoencoder trained only on known examples here.
    train_latent_df  = construct_latent_set(kn_classifier, kn_train, unkn_train)
    val_latent_df = construct_latent_set(kn_classifier, kn_val, unkn_val)#kn_train, unkn_train)
    ## latent_df = construct_latent_set(kn_classifier, kn_val, unkn_val)
    
    # NEXT STEP: Use this latent data to train linear anomaly detector!! :)

    # Note that the train_latent_df is used for determining the initial weight vector
    w, clf = omd(train_latent_df, val_latent_df, anom_classes)

    #with open('weights_oracle_feedback.txt', 'wb') as f:
    #    np.save(f, w)
    # END ELSE

    #latent_df = construct_latent_set(kn_ae, kn_val, unkn_val)
    
    
    # Construct test set and latentify test examples
    kn_unkn_test = construct_latent_set(kn_classifier, kn_test, unkn_test)
    test_target = kn_unkn_test['label']
    kn_unkn_test_trans = loda_transform(clf, kn_unkn_test)
    
    # Test anomaly detection score on linear model
    # plot AUC (start general, then move to indiv classes?)
    scores, y_actual = test_results(kn_unkn_test_trans, w, test_target, anom_classes)
    for i, pred in enumerate(scores):
        print('{}  {}'.format(pred, y_actual[i]))
    # IF BAD, reevaluate LODA initialization

    print('AUROC: {}'.format(roc_auc_score(y_actual, scores)))
    plot_auroc(y_actual, scores)



    
    # NEXT: Run on all 5 anomaly splits.
    
    
    # Use latent space to train classifier AND as input to scoring function for
    # open category detector g
    
if __name__ == '__main__':
    main()

