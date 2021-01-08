import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import os
from pyod.models.loda import LODA
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from oc_data_load import CIFAR10_Data
from vanilla_ae import get_vanilla_ae


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


def omd(data, anom_classes, learning_rate=1e-3):
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
    N = len(data)
    # TODO: Initialize this with LODA values
    theta = get_weight_prior(data)#np.ones(len(data.iloc[0].tolist())-1)
    # Note that this is usually implemented as an
    # online algorithm so number of time steps (steps
    # of this outer loop) are usually ambiguous. Here
    # we have a dataset of some fixed size, so we
    # will start by running a single epoch over the
    # dataset
    # TODO: Mess around with reducing this number
    for i in range(N):
        w = get_nearest_w(theta)
        d_anom, idx = get_max_anomaly_score(w, data)
        y = get_feedback(data.iloc[i]['label'], anom_classes)
        data.drop(data.index[idx])
        # linear loss function
        # loss = -y*np.dot(w,d_anom)
        theta = theta - learning_rate*y*d_anom

    return w


def get_feedback(label, anom_classes):
    '''Checks for membership of label
    in given set of anomaly classes.
    If the instance is anomalous,
    returns 1 else returns -1'''
    y = -1
    if label in anom_classes:
        y = 1
        
    return y

        
def get_max_anomaly_score(w, D):
    '''Returns the element in the dataset
    with the largest anomaly score    '''
    N = len(D)
    D_X = D.drop(columns=['label']).to_numpy()
    x_curr = np.dot(-w, D_X[0])#D_X.iloc[0])
    x_max = x_curr
    for i in range(N):
        x_curr = np.dot(-w, D_X[i])#D_X.iloc[i])
        if x_curr > x_max:
            x_max = x_curr
            idx = i

    return x_max, idx


def get_nearest_w(theta):
    '''Returns the weight vector in the space
    of d-dimensional vectors with positive real 
    numbers that is closest to the given theta'''
    return relu(theta)


def relu(x):
    '''Just makes negative elements 0'''
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


def get_weight_prior(X_val_latent):
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
    learned weights associated with each latent feature. Used as the
    prior in training a linear anomaly detector on all classes
    given latent representation from a model trained on only 6.
    
    '''
    #contamination = 0.4 # 6 known classes, 4 unknown using CIFAR10
    #n_bin = len()
    clf_name = 'LODA'
    threshold = 0.01
    data = X_val_latent.drop(columns=['label'])
    #num_bins = len(data.iloc[0].tolist())
    clf = LODA()#n_bins=num_bins)
    model = clf.fit(data)
    #weight_prior = model.histograms_.mean(axis=0)
    #weight_prior = model.projections_.mean(axis=0)
    #weight_prior[weight_prior < threshold] = 1
    #weight_prior[weight_prior != 1] = 0
    #plt.plot(weight_prior)
    #plt.show()

    num_hists = 100
    num_bins = 10
    hists = clf.histograms_
    projs = clf.projections_
    weight_prior = np.zeros(128)
    max = 0
    for bin_i in range(num_bins):
        for hist in range(num_hists):
            if hists[hist,bin_i] > hists[max,bin_i]:
                max = hist
        weight_prior = np.add(projs[max,:], weight_prior)
    
    return weight_prior

    # y_val_latent_pred = clf.labels_ # binary (0: inlier, 1: outlier)
    # y_train_scores = clf.decision_scores_ # raw outlier scores

    #return clf.get_params() # By default deep=True


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
        latent_batch, _ = model.get_latent(img_batch)
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


def test_results(test_data, weights, anom_classes):
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
    y_hat: numpy array
        Classifications on a per-example basis (+1: anomalous; -1: nominal)

    y: numpy array
        Actual classification of each example  ("" "")
    '''
    num_examples = len(test_data)
    X = test_data.drop(columns=['label'])
    y_class = test_data['label']
    # get_feedback(label, anom_classes)
    y = np.zeros(num_examples)
    y_hat = np.zeros(num_examples)
    for i in range(num_examples):
        y[i] = get_feedback(y_class[i], anom_classes)
    data_iter = tqdm(X.iterrows())
    for i, example in data_iter:
        y_hat[i] = np.dot(-weights, example)

    return y_hat, y
    

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
    kn_ae = get_plain_ae(kn_train, kn_val,'kn_std_ae_split_{}.pth'.format(0))

    ''' <><><><><><><> USE THIS IF YOU NEED AE THAT IS TRAINED ON KN/UNKN <><><><><>
    # Training plain autoencoder on all training data
    kn_unkn_train = torch.utils.data.ConcatDataset([kn_train,unkn_train])
    # This preserves metadata
    # MIGHT NOT NEED kn_unkn_val_frame = pd.concat([kn_val.frame, unkn_val.frame])
    kn_unkn_val = torch.utils.data.ConcatDataset([kn_val,  unkn_val  ])
    kn_unkn_ae  = get_plain_ae(kn_unkn_train, kn_unkn_val,
                              'kn_unkn_std_ae_split_{}.pth'.format(0))
    '''

    if os.path.isfile('weights.txt'):
        # Load weights
        with open('weights.txt', 'rb') as f:
            w = np.load(f)
        
    else:
        # Get latent set used to train linear anomaly detector from the
        # validation set comprised of all classes. Note that we are
        # using the autoencoder trained only on known examples here.
        latent_df = construct_latent_set(kn_ae, kn_val, unkn_val)
    
        # NEXT STEP: Use this latent data to train linear anomaly detector!! :)
        w = omd(latent_df, anom_classes)

        with open('weights.txt', 'wb') as f:
            np.save(f, w)

    # Construct test set and latentify test examples
    kn_unkn_test = construct_latent_set(kn_ae, kn_test, unkn_test)
    
    # Test anomaly detection score on linear model
    # plot AUC (start general, then move to indiv classes?)
    y_hat, y_actual = test_results(kn_unkn_test, w, anom_classes)
    for i, pred in enumerate(y_hat):
        print('{}  {}'.format(pred, y_actual[i]))
    # IF BAD, reevaluate LODA initialization




    
    # NEXT: Run on all 5 anomaly splits.
    
    
    # Use latent space to train classifier AND as input to scoring function for
    # open category detector g
    
if __name__ == '__main__':
    main()
