import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from torchvision import datasets, transforms
from oc_data_load import CIFAR10_Data
from collections import OrderedDict


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


# Classifier to be trained on CIFAR-10.
# Resnet code found at https://github.com/FrancescoSaverioZuppichini/ResNet 

# Implement RESNET-18
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically adding padding based on kernel size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, \
                 conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, \
                    bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

        
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, \
                     stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

        
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, \
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and
    maps the output to the correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(\
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        return self.encoder(x)

    
def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])


'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding_net = nn.Sequential(
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
        )
        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_latent(self, x):
        return self.embedding_net(x)
'''


def train(model, device, tr_data, tr_target, val, val_target, num_epochs=30, learning_rate=1e-3):#batch_size=64 
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    outputs = []
    epoch = 1
    target_val_accuracy = 0.8
    val_accuracy = 0
    val_accuracy_list = []
    # Initial conditions
    val_loss_list = [1E6, 0]
    EPS = 1e-3
    for epoch in range(num_epochs):
    #while val_accuracy < target_val_accuracy:
    #while epoch < num_epochs and abs(val_loss_list[epoch] - val_loss_list[epoch-1]) > EPS:
        for i,img_batch in enumerate(tr_data):
            img_batch = img_batch.to(device)
            batch_target = torch.tensor(tr_target[i]).to(device)
            recon = model(img_batch)
            
            # FIX REPLACE SECOND ARG WITH TARGET VAR
            loss = criterion(recon, batch_target)#img_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch, float(loss)))
        outputs.append((epoch, img_batch, recon),)
        _, val_loss = get_val_loss(model, val, val_target, device)
        val_loss_list.append(val_loss)
        model.eval()
        with torch.no_grad():
            val_accuracy = get_accuracy(model, val, val_target, device)
            val_accuracy_list.append(val_accuracy)
            print('dev acc = {}'.format(val_accuracy))
        model.train()
        #val_accuracy_list.append(val_accuracy)
        epoch += 1
    
        
    return outputs, val_loss_list, val_accuracy_list


def get_accuracy(model, val, val_target, device):
    batch_size = len(val_target[0])
    N = batch_size*len(val_target)
    num_correct = 0
    y_hat = []
    y = []
    for i, img_batch in enumerate(val):
        img_batch = img_batch.to(device)
        #batch_target = torch.tensor(val_target[i]).to(device)
        prediction = model(img_batch)
        prediction = prediction.cpu().numpy()
        for j in range(batch_size):
            y = val_target[i][j]
            y_hat = np.argmax(prediction[j])
            if y_hat == y:
                num_correct += 1
        # print('{} {}'.format(prediction[0], batch_target[0]))
        # if correct
        #     increment num_correct
        # else
        #     move to next

    return num_correct / N


# TODO: Fix this...
def get_val_loss(model, val, val_target, device):
    criterion = nn.CrossEntropyLoss()
    outputs = []
    loss = 0

    # TODO: Update for val. data_loader no longer works here!!!
    for i,img_batch in enumerate(val):
        # Reshape mini-batch data to [N, 32*32] matrix
        # Load it to the active device
        # batch_features = batch_features.view(-1, 32*32).to(device)
        img_batch = img_batch.to(device)
        batch_target = torch.tensor(val_target[i]).to(device)
        # compute reconstructions
        reconstruction = model(img_batch)

        #target = val
        
        # compute cross-entropy loss
        test_loss = criterion(reconstruction, batch_target)

        # add the mini-batch training loss to epoch loss
        loss += test_loss.item()

    # Compute the epoch training loss
    loss = loss / len(val)
    # display the epoch test loss
    print("dev loss = {:.6f}".format(loss))
    outputs.append((None,img_batch,reconstruction),)

    return outputs, loss


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def to_class_index(target_set, classes, split, splits, batch_size):
    N = len(target_set)
    K = len(classes)
    unknown_class_indices = splits[split]
    unknown = [classes[i] for i in unknown_class_indices]
    known = []
    target_indices = []
    for i in range(K):
        if classes[i] not in unknown:
            known.append(classes[i])

    class_dict = dict(zip(known,range(len(known))))

    for i in range(N):
        target_indices.append(class_dict[target_set[i]])

    target_indices = np.array(target_indices)
    target_indices = target_indices.reshape((-1,batch_size))

    #print(val_target)
    #print(target_indices)

    return target_indices


def get_resnet_18_classifier(kn_train=None, kn_val=None, split=0, filename='resnet18_classifier_kn.pth'):
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 30
    NUM_CHANNELS = 3
    NUM_KNOWN = 6
    NUM_UNKNOWN = 4
    BATCH_SIZE = 4
    CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]

    SPLIT = split

    #print('Number of target batches: {}'.format(len(tr_target)))

    #imshow(kn_train.__getitem__(4))
    #print(tr_target[1][0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(NUM_CHANNELS, NUM_KNOWN).to(device)
    
    if os.path.isfile(filename):
        # Load model
        model.load_state_dict(torch.load(filename))
        
    else:
        tr_loader = torch.utils.data.DataLoader(kn_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(kn_val,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

        # Convert labels into lists and drop any labels
        # that belong to any training examples dropped in
        # the DataLoader
        tr_target = kn_train.frame['label'].tolist()
        while len(tr_target) % BATCH_SIZE != 0:
            tr_target.pop()
        val_target = kn_val.frame['label'].tolist()
        while len(val_target) % BATCH_SIZE != 0:
            val_target.pop()

        tr_target = to_class_index(tr_target, CIFAR_CLASSES, SPLIT, splits, BATCH_SIZE)
        val_target = to_class_index(val_target, CIFAR_CLASSES, SPLIT, splits, BATCH_SIZE)
        
        # Train model
        outputs, val_loss, val_acc = train(model, device, tr_loader, tr_target, val_loader, val_target, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), filename)
  
    model.eval()
    return model

    
def main():
    CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

    SPLIT = 0
    NUM_CHANNELS = 3
    NUM_EPOCHS = 7
    NUM_KNOWN = 6
    NUM_UNKNOWN = 4
    BATCH_SIZE = 4
    FILENAME = 'resnet18_classifier_kn.pth'
    
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

    # For split zero, KNOWN = "airplane", "automobile", "bird", "deer", "dog", "truck"

    anom_classes = [CIFAR_CLASSES[i] for i in splits[SPLIT]]

    # Get datasets of known and unknown classes
    kn_train, kn_val, _, _, _, _ = load_data(SPLIT)
    tr_loader = torch.utils.data.DataLoader(kn_train, batch_size=BATCH_SIZE, \
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(kn_val, batch_size=BATCH_SIZE, shuffle=False, \
                                             pin_memory=True)

    tr_target = kn_train.frame['label'].tolist()
    val_target = kn_val.frame['label'].tolist()

    tr_target = to_class_index(tr_target, CIFAR_CLASSES, SPLIT, splits, BATCH_SIZE)
    val_target = to_class_index(val_target, CIFAR_CLASSES, SPLIT, splits, BATCH_SIZE)

    print('Number of target batches: {}'.format(len(tr_target)))

    #imshow(kn_train.__getitem__(4))
    #print(tr_target[1][0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(NUM_CHANNELS, NUM_KNOWN).to(device)
    
    if os.path.isfile(FILENAME):
        # Load model
        model.load_state_dict(torch.jit.load(FILENAME))#load(FILENAME))
        model.eval()
    else:
        # Train model
        outputs, val_loss, val_acc = train(model, device, tr_loader, tr_target,\
                                  val_loader, val_target, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), FILENAME)
    
    plt.plot(val_loss)
    plt.show()
    
        
if __name__ == '__main__':
    main()
