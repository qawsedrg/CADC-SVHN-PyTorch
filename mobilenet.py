import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image
import example_pb2
import nni
from nni.utils import merge_parameter

logger = logging.getLogger('svhn_AutoML')

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.block1=nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU6()
        )
        self.block2=nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU6()
        )
        self.block3=nn.Sequential(nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False), 
                                  nn.BatchNorm2d(out_planes),
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
        
        
class Model(nn.Module):
    #t,c,n,s
    cfg = [[6,  32, 3, 2],
           [6,  64, 4, 2],
           [6,  96, 3, 1],
           [6, 160, 3, 2]]
    #t为扩张系数，c为输出通道数，n为该层重复的次数，s为步长
    def __init__(self,t,c1,c2,c3,c4,c5,n1,n2,n3,n4):
        super().__init__()
        self.cfg[0][1]=c1
        self.cfg[1][1]=c2
        self.cfg[2][1]=c3
        self.cfg[3][1]=c4
        self.cfg[0][2]=n1
        self.cfg[1][2]=n2
        self.cfg[2][2]=n3
        self.cfg[3][2]=n4
        self.cfg[0][0]=t
        self.cfg[1][0]=t
        self.cfg[2][0]=t
        self.cfg[3][0]=t
        self.hidden1 = nn.Sequential(nn.Conv2d(3, self.cfg[0][1], kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.cfg[0][1]),
                                     nn.ReLU())
        self.layers = self._make_layers(in_planes=self.cfg[0][1])
        self.hidden2 = nn.Sequential(nn.Conv2d(self.cfg[3][1], c5, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(c5),
                                     nn.ReLU())
        self._digit11 = nn.Sequential(nn.Linear(c5, 10))
        self._digit21 = nn.Sequential(nn.Linear(c5, 10))
        
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.hidden1(x)
        #print(x.size())
        x = self.layers(x)
        #print(x.size())
        x = self.hidden2(x)
        #print(x.size())
        x = F.avg_pool2d(x, 7)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())

        digit1_logits = self._digit11(x)
        digit2_logits = self._digit21(x)

        return digit1_logits, digit2_logits

def calculloss(digit1_logits, digit2_logits,digits_labels):
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
    loss = digit1_cross_entropy + digit2_cross_entropy
    return loss
    
def train(args,model,device,train_loader,optimizer,epoch):
    for batch_idx, (images, digits_labels) in enumerate(train_loader):
        images, digits_labels = images.to(device), [digit_labels.to(device) for digit_labels in digits_labels]
        digit1_logits, digit2_logits= model.train()(images)
        loss = calculloss(digit1_logits, digit2_logits,digits_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
def test(args,model, device, test_loader):
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (images,digits_labels) in enumerate(test_loader):
            images, digits_labels = images.to(device),[digit_labels.to(device) for digit_labels in digits_labels]
            digit1_logits, digit2_logits= model.eval()(images)

            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]

            num_correct += (digit1_prediction.eq(digits_labels[0]) &
                            digit2_prediction.eq(digits_labels[1])).cpu().sum()

    accuracy = 100*num_correct.item() / len(test_loader.dataset)
    logger.info('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(num_correct, len(test_loader.dataset), accuracy))
    return accuracy
    
class Dataset(data.Dataset):
    def __init__(self, path_to_lmdb_dir, transform):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        with self._reader.begin() as txn:
            value = txn.get(self._keys[index])

        example = example_pb2.Example()
        example.ParseFromString(value)

        image = np.frombuffer(example.image, dtype=np.uint8)
        image = image.reshape([54, 54, 3])
        image = Image.fromarray(image)
        image = self._transform(image)

        length = example.length
        digits = example.digits

        return image, digits

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN MobileNet')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.8, help='SGD momentum (default: 0.8)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='log_interval (default: 100)')
    
    parser.add_argument('--t', type=int, default=6, help='t (default: 6)')
    parser.add_argument('--c1', type=int, default=32, help='c1 (default: 132)')                        
    parser.add_argument('--c2', type=int, default=64, help='c2 (default: 64)')               
    parser.add_argument('--c3', type=int, default=96, help='c3 (default: 96)')                       
    parser.add_argument('--c4', type=int, default=160, help='c4 (default: 160)')                        
    parser.add_argument('--c5', type=int, default=320, help='c5 (default: 320)')                        
    parser.add_argument('--n1', type=int, default=3, help='n1 (default: 3)')                        
    parser.add_argument('--n2', type=int, default=4, help='n2 (default: 4)')                        
    parser.add_argument('--n3', type=int, default=3, help='n3 (default: 3)')                        
    parser.add_argument('--n4', type=int, default=3, help='n4 (default: 3)')
                        
    args, _ = parser.parse_known_args()
    return args
    
def main(args):
    device = torch.device("cuda")
    
    torch.manual_seed(args['seed'])
        
    path_to_train_lmdb_dir = "./data/train.lmdb"
    path_to_val_lmdb_dir = "./data/val.lmdb"
        
    train_transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-45,45)),
                transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.1, hue=.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir, train_transform),
                                                   batch_size=args['batch_size'], shuffle=True)
    test_transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-20,20)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    test_loader = torch.utils.data.DataLoader(Dataset(path_to_val_lmdb_dir, test_transform), batch_size=128, shuffle=False)

    model = Model(args["t"],args["c1"],args["c2"],args["c3"],args["c4"],args["c5"],args["n1"],args["n2"],args["n3"],args["n4"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],momentum=args['momentum'])

    final_test_acc=0
    for epoch in range(1, 15 + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)
        final_test_acc=max(final_test_acc,test_acc)
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')
        
    nni.report_final_result(final_test_acc)
    logger.debug('Final result is %g', final_test_acc)
    logger.debug('Send final result done.')
    
if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
