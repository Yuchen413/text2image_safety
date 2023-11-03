import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# torch
import torch
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader

# global variables
eps = np.finfo(np.float32).eps.item()
torch_cuda = 0


class data_agent():
    # common transformations
    normalize = Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    inv_normalize = Transforms.Normalize(mean=[-(0.485) / 0.229, -(0.456) / 0.224, -(0.406) / 0.225],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    process_PIL = Transforms.Compose([Transforms.Resize((224, 224)),
                                      Transforms.ToTensor(),
                                      normalize])

    def __init__(self, ImageNet_train_dir, ImageNet_val_dir,
                 data_name='ImageNet', train_transform=None, val_transform=None,
                 ):

        self.data_name = data_name
        self.ImageNet_train_dir = ImageNet_train_dir
        self.ImageNet_val_dir = ImageNet_val_dir

        if self.data_name == 'ImageNet':

            if train_transform:
                train_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_train_dir,
                    transform=train_transform,
                )
            else:
                train_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_train_dir,
                    transform=Transforms.Compose([
                        Transforms.RandomResizedCrop(224),
                        Transforms.RandomHorizontalFlip(),
                        Transforms.ToTensor(),
                        self.normalize,
                    ])
                )

            if val_transform:
                val_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_val_dir,
                    transform=val_transform,
                )
            else:
                val_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_val_dir,
                    transform=Transforms.Compose([
                        Transforms.Resize(256),
                        Transforms.CenterCrop(224),
                        Transforms.ToTensor(),
                        self.normalize,
                    ])
                )

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            # easy to update the loaders and save memory
            self.train_loader = None
            self.val_loader = None

            print('Your {} dataset has been prepared, please remember to update the loaders with the batch size'
                  .format(self.data_name))

    def update_loaders(self, batch_size):

        self.batch_size = batch_size

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

        # use del for safety
        del self.train_loader
        self.train_loader = train_loader
        del self.val_loader
        self.val_loader = val_loader
        print('Your {0} dataloaders have been updated with batch size {1}'
              .format(self.data_name, self.batch_size))

    def get_indices(self, label, save_dir, correct=False, cnn=None,
                    train=True, process_PIL=process_PIL):
        '''
        input:
        label: int
        correct: flag to return the indices of the data point which is crrectly classified by the cnn
        cnn: pytorch model
             [old]model name, which model to use to justify whether the data points are correclty classified
             [old]change from string to torch model in the function
        process_PIL: transform used in the 'correct' mode
        return:
        torch.tensor containing the indices in the self.train_dataset or self.val_dataset,
        or custom dataset when in 'correct' mode
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, 'label_{}_train-set_{}_correct_{}.pt'.format(label, train, correct))
        if os.path.exists(file_name):
            indices = torch.load(file_name)
            return indices
        else:
            if train:
                targets_tensor = torch.Tensor(self.train_dataset.targets)
            else:
                targets_tensor = torch.Tensor(self.val_dataset.targets)

            temp = torch.arange(len(targets_tensor))
            indices = temp[targets_tensor == label]

            if correct:
                cnn = cnn.cuda(torch_cuda).eval()
                if train:
                    temp_dataset = Datasets.ImageFolder(
                        root=self.ImageNet_train_dir,
                        transform=process_PIL,
                    )
                else:
                    temp_dataset = Datasets.ImageFolder(
                        root=self.ImageNet_val_dir,
                        transform=process_PIL,
                    )
                with torch.no_grad():
                    wrong_set = []
                    label_tensor = torch.Tensor([label]).long().cuda(torch_cuda)
                    for index in indices:
                        input_tensor = temp_dataset.__getitem__(index)[0]
                        input_tensor = input_tensor.cuda(torch_cuda).unsqueeze(0)
                        output_tensor = cnn(input_tensor)
                        if output_tensor.argmax() != label_tensor:
                            wrong_set.append(index)
                    for item in wrong_set:
                        indices = indices[indices != item]
            torch.save(indices, file_name)
            return indices

    @staticmethod
    def show_image_from_tensor(img, inv=False, save_dir=None, dpi=300, tight=True):
        '''
        inv: flag to recover the nomalization transformation on images from ImageNet
        '''

        if img.dim() == 4:
            assert img.size(0) == 1, 'this function currently supports showing single image'
            img = img.squeeze(0)
            print('The batch dimension has been squeezed')

        if inv:
            img = data_agent.inv_normalize(img)

        npimg = img.cpu().numpy()
        # fig = plt.figure(figsize = (5, 15))
        fit = plt.figure()
        if len(npimg.shape) == 2:
            print('It is a gray image')
            plt.imshow(npimg, cmap='gray')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()

        if save_dir is not None:
            if tight:
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(fname=save_dir,
                        dpi=dpi, facecolor='w', edgecolor='w', format='png')

    @staticmethod
    def save_with_content(path, image, dpi=300):
        '''
        image: numpy image with shape (h, w, c)
        '''
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image)
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    '''
    This function comes from 
    https://github.com/bearpaw/pytorch-classification/blob/master/utils/eval.py
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res