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
#from vanilla_ae import get_vanilla_ae
from classifier import get_resnet_18_classifier
from oracle_omd import load_data, construct_latent_set, get_feedback


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
    kn_classifier = get_resnet_18_classifier(kn_train, kn_val)

    ''' <><><><><><><> USE THIS IF YOU NEED AE THAT IS TRAINED ON KN/UNKN <><><><><>
    # Training plain autoencoder on all training data
    kn_unkn_train = torch.utils.data.ConcatDataset([kn_train,unkn_train])
    # This preserves metadata
    # MIGHT NOT NEED kn_unkn_val_frame = pd.concat([kn_val.frame, unkn_val.frame])
    kn_unkn_val = torch.utils.data.ConcatDataset([kn_val,  unkn_val  ])
    kn_unkn_ae  = get_plain_ae(kn_unkn_train, kn_unkn_val,
                              'kn_unkn_std_ae_split_{}.pth'.format(0))

    
    if os.path.isfile('weights_no_oracle.txt'):
        # Load weights
        with open('weights_no_oracle.txt', 'rb') as f:
            w = np.load(f)
        
    else:
        # Get latent set used to train linear anomaly detector from the
        # validation set comprised of all classes. Note that we are
        # using the autoencoder trained only on known examples here.
        latent_df = construct_latent_set(kn_ae, kn_val, unkn_val)
    
        # NEXT STEP: Use this latent data to train linear anomaly detector!! :)
        w = omd(latent_df, anom_classes)

        with open('weights_oracle_feedback.txt', 'wb') as f:
            np.save(f, w)

    '''

    # Constructs latent dataset comprised of known and unknown examples
    latent_df_train = construct_latent_set(kn_classifier, kn_val, unkn_val)

    latent_train_X = latent_df_train.drop(columns=['label'])
    clf = LODA(n_bins=len(latent_train_X)//50, n_random_cuts=26000)
    loda_model = clf.fit(latent_train_X)
    
    # Construct test set and latentify test examples (mixed known and unknown)
    latent_df_test = construct_latent_set(kn_classifier, kn_test, unkn_test)
    latent_test_X = latent_df_test.drop(columns=['label'])
    y_labels = latent_df_test['label']
    y_actual = np.zeros(len(y_labels))
    y_hat = -clf.decision_function(latent_test_X)

    for i, label in enumerate(y_labels):
        # TODO: -CHECK get_feedback TO ENSURE CORRECTNESS
        #       -Also, try this experiment one more time
        #       -Also, triple check that anomalous data is being pre-processed
        #        correctly. Maybe utilize imshow?
        y_actual[i] = get_feedback(label, anom_classes)
    
    # Test anomaly detection score on linear model
    # plot AUC (start general, then move to indiv classes?)
    #y_hat, y_actual = test_results(kn_unkn_test, w, anom_classes)
    for i, pred in enumerate(y_hat):
        print('{}  {}'.format(pred, y_actual[i]))
    # IF BAD, reevaluate LODA initialization

    print('AUROC: {}'.format(roc_auc_score(y_actual, y_hat)))
    fpr, tpr, thresholds = roc_curve(y_actual, y_hat, pos_label=1)
    print('fpr: {}'.format(fpr))
    print('tpr: {}'.format(tpr))
    plt.plot(fpr,tpr)
    plt.show()
    print(loda_model.histograms_)
    
    # NEXT: Run on all 5 anomaly splits.
    
    
    # Use latent space to train classifier AND as input to scoring function for
    # open category detector g
    
if __name__ == '__main__':
    main()
