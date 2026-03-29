from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
from itertools import count
import time
import random
import numpy as np

from models.models import *
from models.preact_resnet import *

from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# == parser start
parser = argparse.ArgumentParser(description="PyTorch")
# base setting 1: fixed
parser.add_argument("--job-id", type=int, default=1)
parser.add_argument("--seed", type=int, default=5)
# base setting 2: fixed
parser.add_argument("--test-batch-size", type=int, default=100)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--data-path", type=str, default="./dataset/")
# experiment setting
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--data-aug", type=int, default=0)
parser.add_argument("--model", type=str, default="LeNet")
# method setting
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--ssize", type=int, default=64)
parser.add_argument("--method", type=int, default=0)
# --method=0: standard
# --method=1: q-SGD
args = parser.parse_args()
# == parser end
data_path = args.data_path + args.dataset
if not os.path.isdir(data_path):
    os.makedirs(data_path)

result_path = (
    f"./results/{args.dataset}_{args.model}_" f"method{args.method}_bs{args.batch_size}"
)

if args.method == 1:
    result_path += f"_ssize{args.ssize}"

result_path += f"_job{args.job_id}"
filep = open(result_path + ".txt", "w")

out_str = str(args)
print(out_str)
filep.write(out_str + "\n")

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

out_str = "initial seed = " + str(args.seed)
print(out_str)
filep.write(out_str + "\n\n")

# ===============================================================
# === dataset setting
# ===============================================================
kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_Sampler = None
test_Sampler = None
Shuffle = True
if args.dataset == "mnist":
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomAffine(15, scale=(0.85, 1.15)),
                transforms.ToTensor(),
            ]
        )
    train_data = datasets.MNIST(
        data_path, train=True, download=True, transform=train_transform
    )
    test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=test_transform
    )
elif args.dataset == "cifar10":
    nh = 32
    nw = 32
    nc = 3
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    train_data = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=test_transform
    )
elif args.dataset == "cifar100":
    nh = 32
    nw = 32
    nc = 3
    num_class = 100
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=test_transform
    )
elif args.dataset == "svhn":
    nh = 32
    nw = 32
    nc = 3
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.SVHN(
        data_path, split="train", download=True, transform=train_transform
    )
    test_data = datasets.SVHN(
        data_path, split="test", download=True, transform=test_transform
    )
elif args.dataset == "fashionmnist":
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 20
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    train_data = datasets.FashionMNIST(
        data_path, train=True, download=True, transform=train_transform
    )
    test_data = datasets.FashionMNIST(
        data_path, train=False, download=True, transform=test_transform
    )
elif args.dataset == "kmnist":
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [transforms.RandomCrop(28, padding=2), transforms.ToTensor()]
        )
    train_data = datasets.KMNIST(
        data_path, train=True, download=True, transform=train_transform
    )
    test_data = datasets.KMNIST(
        data_path, train=False, download=True, transform=test_transform
    )
elif args.dataset == "semeion":
    nh = 16
    nw = 16
    nc = 1
    num_class = 10  # the digits from 0 to 9 (written by 80 people twice)
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(16, padding=1),
                transforms.RandomAffine(4, scale=(1.05, 1.05)),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.SEMEION(data_path, transform=train_transform, download=True)
    test_data = train_data
    random_index = np.load(data_path + "/random_index.npy")
    train_size = 1000
    train_Sampler = SubsetRandomSampler(random_index[range(train_size)])
    test_Sampler = SubsetRandomSampler(random_index[range(train_size, len(test_data))])
    Shuffle = False
elif args.dataset == "fakedata":
    nh = 24
    nw = 24
    nc = 3
    num_class = 10
    end_epoch = 50
    train_size = 1000
    test_size = 1000
    train_data = datasets.FakeData(
        size=train_size + test_size,
        image_size=(nc, nh, nw),
        num_classes=num_class,
        transform=train_transform,
    )
    test_data = train_data
    train_Sampler = SubsetRandomSampler(range(train_size))
    test_Sampler = SubsetRandomSampler(range(train_size, len(test_data)))
    Shuffle = False
else:
    print("specify dataset")
    exit()
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    sampler=train_Sampler,
    shuffle=Shuffle,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.test_batch_size,
    sampler=test_Sampler,
    shuffle=False,
    **kwargs,
)

# ===============================================================
# === model setting
# ===============================================================
if args.model == "LeNet":
    model = LeNet(nc, nh, nw, num_class).to(device)
elif args.model == "PreActResNet18":
    model = PreActResNet18(nc, num_class).to(device)
elif args.model == "Linear" or args.model == "SVM":
    dx = nh * nw * nc
    model = Linear(dx, num_class).to(device)
else:
    print("specify model")
    exit()


# ===============================================================
# === utils def
# ===============================================================
def select_training_loss(cr_loss, method, ssize):
    bs = cr_loss.size(0)

    # standard SGD
    if method == 0:
        return torch.mean(cr_loss)

    # ordered SGD: https://arxiv.org/abs/1907.04371
    elif method == 1:
        if ssize >= bs:
            return torch.mean(cr_loss)
        return torch.topk(cr_loss, k=min(ssize, bs))[0].mean()

    # kl-dro/exponential weighting: https://arxiv.org/abs/1610.03425/ (not sure) (change to softmax? in the works)
    elif method == 2:
        tau = 1.0  # hyperparameter
        weights = torch.softmax(cr_loss / tau, dim=0)
        return torch.sum(weights * cr_loss)

    # z-score weighting: ours
    elif method == 3:
        mean = cr_loss.mean()
        std = cr_loss.std() + 1e-8
        z = (cr_loss - mean) / std
        weights = torch.relu(z)
        weights = weights / (weights.sum() + 1e-8)
        return torch.sum(weights * cr_loss)

    # focal weighting: idea, basically weighting harder samples more
    elif method == 5:
        gamma = 2.0
        weights = cr_loss**gamma
        weights = weights / (weights.sum() + 1e-8)
        return torch.sum(weights * cr_loss)
    
    elif method == 6:  # rank-based
        alpha = 2.0
        bs = cr_loss.size(0)
        ranks = torch.argsort(torch.argsort(cr_loss)) + 1
        weights = ranks.float() ** alpha
        weights = weights / weights.sum()
        return torch.sum(weights * cr_loss)


    else:
        raise ValueError(f"Unknown method: {method}")


def lr_decay_func(optimizer, lr_decay=0.1):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= 0.1
    return optimizer


def lr_scheduler(optimizer, epoch, lr_decay=0.1, interval=10):
    if args.data_aug == 0:
        if epoch == 10 or epoch == 50:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)
    if args.data_aug == 1:
        if epoch == 10 or epoch == 100:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)
    return optimizer


class multiClassHingeLoss(nn.Module):
    def __init__(self):
        super(multiClassHingeLoss, self).__init__()

    def forward(self, output, y):
        index = torch.arange(0, y.size()[0]).long().to(device)
        output_y = output[index, y].view(-1, 1)
        loss = output - output_y + 1.0
        loss[index, y] = 0
        loss[loss < 0] = 0
        loss = torch.sum(loss, dim=1) / output.size()[1]
        return loss


hinge_loss = multiClassHingeLoss()

# ===============================================================
# === train optimization def
# ===============================================================
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
ssize = args.ssize


def train(epoch):
    global optimizer, ssize
    model.train()
    optimizer = lr_scheduler(optimizer, epoch)

    train_acc_prev = pl_result[epoch - 1, 0, 1]
    if train_acc_prev >= 99.5 and ssize > 4:
        ssize = 4
        optimizer = lr_decay_func(optimizer, lr_decay=0.5)
    elif train_acc_prev >= 95 and ssize > 8:
        ssize = 8
    elif train_acc_prev >= 90 and ssize > 16:
        ssize = 16
    elif train_acc_prev >= 80 and ssize > 32:
        ssize = 32

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        h1 = model(x)
        if args.model == "SVM":
            cr_loss = hinge_loss(h1, y)
        else:
            cr_loss = F.cross_entropy(h1, y, reduction="none")

        loss = select_training_loss(cr_loss, args.method, ssize)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ===============================================================
# === train/test output def
# ===============================================================
def output(data_loader):
    if data_loader == train_loader:
        model.train()
    elif data_loader == test_loader:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_size = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            h1 = model(x)
            y_hat = h1.argmax(dim=1)
            if args.model == "SVM":
                total_loss += torch.mean(hinge_loss(h1, y)).item() * y.size(0)
            else:
                total_loss += F.cross_entropy(h1, y).item() * y.size(0)
            total_correct += y_hat.eq(y).sum().item()
            total_size += y.size(0)

    total_loss /= total_size
    total_acc = 100.0 * total_correct / total_size
    return total_loss, total_acc


# ===============================================================
# === start computation
# ===============================================================
# == for plot
pl_result = np.zeros(
    (end_epoch + 1, 3, 2)
)  # epoch * (train, test, time) * (loss , acc)
# == main loop start
time_start = time.time()
for epoch in count(0):
    if epoch >= 1:
        train(epoch)

    train_loss, train_acc = output(train_loader)
    test_loss, test_acc = output(test_loader)

    pl_result[epoch, 0, :] = (train_loss, train_acc)
    pl_result[epoch, 1, :] = (test_loss, test_acc)

    time_current = time.time() - time_start
    pl_result[epoch, 2, 0] = time_current
    np.save(result_path + "_pl", pl_result)

    if args.method == 1:
        out_str = (
            f"Epoch {epoch} | "
            f"q={ssize} | "
            f"tr_loss={train_loss:.3f} tr_acc={train_acc:.2f} | "
            f"te_loss={test_loss:.3f} te_acc={test_acc:.2f} | "
            f"time={time_current:.1f}"
        )
    else:
        out_str = (
            f"Epoch {epoch} | "
            f"tr_loss={train_loss:.3f} tr_acc={train_acc:.2f} | "
            f"te_loss={test_loss:.3f} te_acc={test_acc:.2f} | "
            f"time={time_current:.1f}"
        )

    print(out_str)
    filep.write(out_str + "\n")

    if epoch == end_epoch:
        break
