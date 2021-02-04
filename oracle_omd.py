import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import os
import math
from sklearn.linear_model import LogisticRegression
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
    T = N
    labels = latent_data['label']
    theta, clf = get_weight_prior(train_latent_data)
    data = loda_transform(clf, latent_data)
    #print('Weight Prior: {}'.format(theta))
    # Note that this is usually implemented as an
    # online algorithm so number of time steps (steps
    # of this outer loop) are usually ambiguous. Here
    # we have a dataset of some fixed size, so we
    # will start by running a single epoch over the
    # dataset
    for i in range(T):
        w = get_nearest_w(theta)
        #w = theta
        d_anom, idx = get_max_anomaly_score(w, data)
        y = get_feedback(labels.iloc[idx], anom_classes)
        # TAG
        #data.drop(data.index[idx])
        np.delete(data, idx)
        # linear loss function
        # loss = -y*np.dot(w,d_anom)
        #print('y: {}'.format(y))
        # TODO: Let sign be positive
        theta = theta + learning_rate*y*data[idx]#d_anom#- learning_rate*y*d_anom
        #print('theta: {}'.format(theta))
        #print(i)
    #print('w: {}'.format(w))
    print("OMD Done")
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
    num_bins  = len(hists[0])
    data = X.to_numpy()
    #transformed_data = np.zeros(N)
    transformed_data = []
    for i in range(N):
        # Create a copy that we can modify to yield
        # the ith training example
        ith_hists = np.copy(hists)
        for j in range(num_hists):
            wj = 0
            # dim(projections_[j,:]) is 512, dim(data[i]) is 513...
            # For some reason the dimension of data here is 1 larger than
            # the examples provided when loda model was trained... WHY?
            #print("Length of data[i]: {}".format(len(data[i])))
            #print("Length of projections_[j,:]: {}".format(len(loda_clf.projections_[j,:])))
            #print(data[i])
            #print('proj 0: {}'.format(loda_clf.projections_[j,0]))
            #print('proj 512: {}'.format(loda_clf.projections_[j,512]))
            # This is just an index for the example number (not a useful feature)
            #print('data 0: {}'.format(data[i,0]))
            #print('data 512: {}'.format(data[i,512])) 
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
        #print('x_curr: {}'.format(x_curr))
        if x_curr > x_max:
            #print('x_curr is new x_max!------------------------------------------------------')
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


def get_weight_prior(X_latent):
    '''
    Get weight vector prior using LODA (Pevny16)
    
    Parameters
    ----------
    X_val_latent : numpy array
        Describes the latent representation of images in the
        validation set. Note: This contains known training examples.

    Returns
    -------
    ndarray: Concatenated one-hot histogram vectors, where a given bin
    is a 1 if it has the greatest probability. Used as the
    prior in training a linear anomaly detector on all classes
    given latent representation from a model trained on only 6.
    object: Fitted LODA estimator.
    '''
    X = X_latent.drop(columns=['label'])
    N = len(X)
    n_bins = 10
    n_random_proj = 100
    clf = LODA(n_bins=n_bins, n_random_cuts=n_random_proj)
    model = clf.fit(X)
    # Note: Laplace smoothing is done when histograms are
    # created inside of pyod
    hists = model.histograms_
    weight_prior = np.copy(model.histograms_)
    #print(weight_prior)
    #weight_prior = weight_prior * 1e-12
    # Correcting laplace smoothing in pyod LODA
    print(weight_prior.shape)
    weight_prior = weight_prior*(N+1e-12)-1e-12
    print(weight_prior.shape)
    # Redoing the Laplace smoothing
    for i in range(n_random_proj):
        weight_prior[i,:] += 1
        weight_prior[i,:] /= np.sum(weight_prior[i,:])
    weight_prior = -np.log2(weight_prior)
    
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
        # Can this be verified?
        concat_data.append([dataset[i], df.iloc[i]['label']])
         
    return concat_data


def construct_latent_set(model, kn_dataset, unkn_dataset=None):
    '''Build dataset from latent representation given
    a model that acts as the encoder and a dataset of
    raw data that is transformed by the encoder'''
    kn_X_y = concat_design_and_target(kn_dataset)#, metadata)
   
    # If unkn_dataset is specified, then concatenate the two 
    if unkn_dataset:
        unkn_X_y = concat_design_and_target(unkn_dataset)
        kn_unkn_X_y = torch.utils.data.ConcatDataset([kn_X_y, unkn_X_y])
        dataset = kn_unkn_X_y
    else:
        dataset = kn_X_y

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    col_labels_loaded = False
    col_labels = []
    embed_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader = tqdm(loader)
    
    # NOTE: Each image batch consists of one image
    for i, (img_batch, label) in enumerate(loader):
        #if i < 10:
        #    filename = '{}_{}.png'.format(label[0],i)
        #    img = img_batch[0]
        #    img_np = img.numpy()
        #    img_np = np.swapaxes(img_np, 0, 2)
        #    plt.imshow(img_np)
        #    plt.savefig(filename)
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
    #print(val_latent_rep_df)
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
        # TODO: MAKE WEIGHTS POSITIVE AGAIN!!!!
        scores[i] = np.dot(weights, example)#-weights, example)

    return scores, y


def plot_auroc(y_actual, scores):
    fpr, tpr, thresholds = roc_curve(y_actual, scores, pos_label=1)
    plt.plot(fpr,tpr)
    plt.savefig('auroc_plot.png')


def main():
    CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

    SPLIT = 0
    
    MODEL_DATA_DIRECTORY = '/nfs/hpc/share/noelt/data/Oracle_Data' 

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
    
    NUM_SPLITS = 3#len(splits)

    # Temporary, had to specify last splits because all 5 produce files
    # that are collectively too big for my home folder in the hpc
    for j in range(NUM_SPLITS): 
        
         # To revert, replace j with SPLIT
         anom_classes = [CIFAR_CLASSES[i] for i in splits[SPLIT]]
         # DEBUG
         #print(anom_classes)
         # GUBED
         # Get datasets of known and unknown classes
         # To revert, replace j with SPLIT
         
         kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test = load_data(j) 

         model_path = os.path.join(MODEL_DATA_DIRECTORY, 'resnet18_classifier_kn_{}.pth'.format(j))
         kn_classifier = get_resnet_18_classifier(kn_train, kn_val, split=j, filename=model_path)
         
         '''
         # Load latent representations of the examples from the nominal classes 
         # in the training set
         if os.path.isfile('train_latent_df.csv'):
             train_latent_df = pd.read_csv('train_latent_df.csv')
         else:
             train_latent_df  = construct_latent_set(kn_classifier, kn_train)
             train_latent_df.to_csv('train_latent_df.csv', index=False)
     	
         # Load latent representations of all examples in the validation set
         if os.path.isfile('val_latent_df.csv'):
             val_latent_df = pd.read_csv('val_latent_df.csv')
         else:
             val_latent_df = construct_latent_set(kn_classifier, kn_val, unkn_val)
             val_latent_df.to_csv('val_latent_df.csv', index=False)
     	
         # Note that the train_latent_df is used for determining the initial weight vector
         w, clf = omd(train_latent_df, val_latent_df, anom_classes)
     	
         # Construct test set and embed test examples
         if os.path.isfile('test_latent_df.csv'):
             test_latent_df = pd.read_csv('test_latent_df.csv')
         else:   
             test_latent_df = construct_latent_set(kn_classifier, kn_test, unkn_test)
             test_latent_df.to_csv('test_latent_df.csv', index=False)
         '''
         
         Z_train_filename = os.path.join(MODEL_DATA_DIRECTORY, 'train_latent_df_{}.csv'.format(j)) 
         Z_val_filename   = os.path.join(MODEL_DATA_DIRECTORY, 'val_latent_df_{}.csv'.format(j))
         Z_test_filename  = os.path.join(MODEL_DATA_DIRECTORY, 'test_latent_df_{}.csv'.format(j))
         
     	 
         # Load latent representations of the examples from the nominal classes 
         # in the training set
         if os.path.isfile(Z_train_filename):
             train_latent_df = pd.read_csv(Z_train_filename)
         else:
             train_latent_df  = construct_latent_set(kn_classifier, kn_train)
             train_latent_df.to_csv(Z_train_filename, index=False)
     	
         # Load latent representations of all examples in the validation set
         if os.path.isfile(Z_val_filename):
             val_latent_df = pd.read_csv(Z_val_filename)
         else:
             val_latent_df = construct_latent_set(kn_classifier, kn_val, unkn_val)
             val_latent_df.to_csv(Z_val_filename, index=False)
     	
         # Note that the train_latent_df is used for determining the initial weight vector
         w, clf_omd = omd(train_latent_df, val_latent_df, anom_classes)
     	
         # Construct test set and embed test examples
         if os.path.isfile(Z_test_filename):
             test_latent_df = pd.read_csv(Z_test_filename)
         else:   
             test_latent_df = construct_latent_set(kn_classifier, kn_test, unkn_test)
             test_latent_df.to_csv(Z_test_filename, index=False)
     	
             
         # Logistic regression test
         X_val = val_latent_df.drop(columns=['label'])
         X_val = X_val.values
         y_val = val_latent_df['label']
         len_val = len(y_val)
         for i in range(len_val):
             y_val.iloc[i] = get_feedback(y_val.iloc[i], anom_classes)
         y_val = y_val.values
         y_val = y_val.astype('int')
         clf = LogisticRegression(max_iter=1000).fit(X_val, y_val)
     	
         # Calculating logistic regression accuracy
         X_test = test_latent_df.drop(columns=['label'])
         X_test = X_test.values
         y_test = test_latent_df['label']
         len_test = len(y_test)
         for i in range(len_test):
             y_test.iloc[i] = get_feedback(y_test.iloc[i], anom_classes)
         y_test = y_test.values
         y_test = y_test.astype('int')
         # TODO binarize labels
         logistic_score = clf.score(X_test, y_test)
         print('binary logistic regression score, split {}: {}\n'.format(j, logistic_score), 
               file=open("results.txt", "a+"))
         
         # TODO: Compute Binary Logistic regression scores on LODA transforms
         # Put validation data through loda transform, then pass it to
         # regression, then score the classifier on the LODA transformed test
         # data below.
        
         # Specify path to save loda-transformed latent representation of the validation set
         loda_tx_val_filename = os.path.join(MODEL_DATA_DIRECTORY, 'val_loda_tx_latent_{}.npy'.format(j)) 
 
         if os.path.isfile(loda_tx_val_filename):
             with open(loda_tx_val_filename, 'rb') as f:
                 kn_unkn_val_loda_tx = np.load(f)
         else:
             kn_unkn_val_loda_tx = loda_transform(clf_omd, val_latent_df)
             np.save(loda_tx_val_filename, kn_unkn_val_loda_tx) 
         
         # Train a binary logistic regression classifier on the loda-tx'd latent representations    
         clf_loda_repr       = LogisticRegression(max_iter=1000).fit(kn_unkn_val_loda_tx, y_val)
         
         # Specify path to save loda-transformed latent representation of the test set
         loda_tx_test_filename = os.path.join(MODEL_DATA_DIRECTORY, 'val_loda_tx_test_{}.npy'.format(j)) 
 
         if os.path.isfile(loda_tx_test_filename):
             with open(loda_tx_test_filename, 'rb') as f:
                 kn_unkn_test_loda_tx = np.load(f)
         else:
             kn_unkn_test_loda_tx = loda_transform(clf_omd, test_latent_df)
             np.save(loda_tx_test_filename, kn_unkn_test_loda_tx) 
         
         # See how the classifier performs
         loda_tx_logistic_score = clf_loda_repr.score(kn_unkn_test_loda_tx, y_test)
         print('loda tx binary logistic regression score, split {}: {}\n'.format(j, logistic_score), 
               file=open("results.txt", "a+"))

         
         # Test anomaly detection score on linear model
         # plot AUC (start general, then move to indiv classes?)
         test_target          = test_latent_df['label']
         scores, y_actual = test_results(kn_unkn_loda_tx, w, test_target, anom_classes)
         #for i, pred in enumerate(scores):
             #print('{}  {}'.format(pred, y_actual[i]))
         # IF BAD, reevaluate LODA initialization
         print(y_actual)
         print('AUROC_{}: {}\n'.format(j, roc_auc_score(y_actual, scores)), file=open("results.txt", "a+"))
         #plot_auroc(y_actual, scores)
    
    # NEXT: Run on all 5 anomaly splits.
    
    # Use latent space to train classifier AND as input to scoring function for
    # open category detector g
   
 
if __name__ == '__main__':
    main()
