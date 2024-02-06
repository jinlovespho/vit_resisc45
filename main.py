import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
import warmup_scheduler
from PIL import Image, ImageEnhance, ImageOps
from torchvision.models import VisionTransformer, ViT_B_16_Weights, vit_b_16
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from torchgeo import datasets
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train', type=str)
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument("--dataset", default='resisc45', type=str,
                    help='c100 or imagenet or imagenet-small or stanfordcars or food101 or flowers-102 or oxford-iiit-pet or caltech101 or caltech256 or resisc45')
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--epoch", default=75, type=int)
parser.add_argument("--warmup", default=10, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--eps", default=1e-6, type=float)
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--alpha", default=0.2, type=float)
parser.add_argument("--randaugment", action="store_true")
parser.add_argument("--num_ops", default=2, type=int)
parser.add_argument("--magnitude", default=10, type=int)
parser.add_argument("--batch", default=64, type=int)
parser.add_argument("--val_batch", default=1024, type=int)
parser.add_argument("--patch", default=16, type=int)
parser.add_argument("--img_size", type=int)
parser.add_argument("--crop_size", type=int)
parser.add_argument("--model_name", default='ViT-small', type=str, help='ViT-tiny,small,base,large,huge')
parser.add_argument("--num_layers", default=0, type=int)
parser.add_argument("--num_heads", default=0, type=int)
parser.add_argument("--hidden_dim", default=0, type=int)
parser.add_argument("--mlp_dim", default=0, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--num_classes", type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project_name", default='Resic45-ViT', type=str)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def main():
    if args.dataset=='c100':
        total_img_num = 50000
        args.num_classes = 100
        args.img_size = 32
        args.crop_size = 32
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.ToTensor(), transforms.ToTensor()
        if args.mode=='train':
            data_train = torchvision.datasets.CIFAR100(root='/home/hyukju/dataset/CIFAR100/',train=True, transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.CIFAR100(root='/home/hyukju/dataset/CIFAR100/',train=False, transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='imagenet-small':
        total_img_num = 15358
        args.num_classes = 12
        args.img_size = 256
        args.crop_size = 256
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]), transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        if args.mode=='train':
            data_train = torchvision.datasets.ImageNet(root='/home/hyukju/dataset/ImageNet2012-small/',split='train', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.ImageNet(root='/home/hyukju/dataset/ImageNet2012-small/',split='val', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='imagenet':
        total_img_num = 1281167
        args.num_classes = 1000
        args.img_size = 256
        args.crop_size = 256
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]), transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        if args.mode=='train':
            data_train = torchvision.datasets.ImageNet(root='/home/hyukju/dataset/ImageNet2012/',split='train', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.ImageNet(root='/home/hyukju/dataset/ImageNet2012/',split='val', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='stanfordcars':
        total_img_num = 8145
        args.num_classes = 196
        args.img_size = 256
        args.crop_size = 256
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]), transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        if args.mode=='train':
            data_train =  torchvision.datasets.StanfordCars(root='/home/hyukju/dataset/',split='train', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.StanfordCars(root='/home/hyukju/dataset/',split='test', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='food101':
        total_img_num = 75750
        args.num_classes = 101
        args.img_size = 512
        args.crop_size = 512
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.ToTensor(), transforms.ToTensor()
        if args.mode=='train':
            data_train = torchvision.datasets.Food101(root='/home/hyukju/dataset/',split='train', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.Food101(root='/home/hyukju/dataset/',split='test', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='flowers-102':
        total_img_num = 6150
        args.num_classes = 102
        args.img_size = 592
        args.crop_size = 592
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False), transforms.ToTensor()]), transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False), transforms.ToTensor()])
            # train_transform, test_transform = transforms.ToTensor(), transforms.ToTensor()
        if args.mode=='train':
            data_train = torchvision.datasets.Flowers102(root='/home/hyukju/dataset/', split='test', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.Flowers102(root='/home/hyukju/dataset/', split='val', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='oxford-iiit-pet':
        total_img_num = 3680
        args.num_classes = 37
        args.img_size = 224
        args.crop_size = 224
        if args.autoaugment:
            train_transform, test_transform = get_transform(args)
        else:
            train_transform, test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]), transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        if args.mode=='train':
            data_train = torchvision.datasets.OxfordIIITPet(root='/home/hyukju/dataset/', split='trainval', transform=train_transform)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  torchvision.datasets.OxfordIIITPet(root='/home/hyukju/dataset/', split='test', transform=test_transform)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='caltech101':
        total_img_num = 7809
        args.num_classes = 101
        args.img_size = 256
        args.crop_size = args.img_size
        if args.autoaugment:
            transform, _= get_transform(args)
        else:
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        dataset = torchvision.datasets.Caltech101(root='/home/hyukju/dataset/', transform=transform)
        total = len(dataset)
        train = int(total*0.9)
        val = total - train
        data_train, data_val = random_split(dataset, [train, val])
        if args.mode=='train':
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    elif args.dataset=='caltech256':
        total_img_num = 27546
        args.num_classes = 257
        args.img_size = 256
        args.crop_size = args.img_size
        if args.autoaugment:
            print('autoaugment impossible!')
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
        dataset = torchvision.datasets.Caltech256(root='/home/hyukju/dataset/', transform=transform)
        total = len(dataset)
        train = int(total*0.9)
        val = total - train
        data_train, data_val = random_split(dataset, [train, val])
        if args.mode=='train':
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    
    # 실행  
    elif args.dataset=='resisc45':
        total_img_num = 18900
        args.num_classes = 45
        args.img_size = 256
        args.crop_size = 256
        if args.randaugment:
            transform = RandAugment_
        else:
            transform = None
        if args.mode=='train':
            data_train = datasets.RESISC45(root=args.data_path, split='train', transforms=transform, download=True)
            train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
        data_val =  datasets.RESISC45(root=args.data_path, split='test', transforms=None, download=True)
        val_loader = DataLoader(data_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers)
    
    experiment_name = f"{args.model_name}_batch:{args.batch}_lr:{args.lr}_wd:{args.weight_decay}_dropout:{args.dropout}_warmup:{args.warmup}_alpha:{args.alpha}_numops:{args.num_ops}_magnitude:{args.magnitude}"
    wandb.init(project=args.project_name)
    wandb.run.name = experiment_name
    wandb.run.save()
    if args.model_name=='ViT-tiny':
        args.num_layers = 12
        args.num_heads = 3
        args.hidden_dim = 192
        args.mlp_dim = 768
    elif args.model_name=='ViT-small':
        args.num_layers = 12
        args.num_heads = 6
        args.hidden_dim = 384
        args.mlp_dim = 1536
    elif args.model_name=='ViT-base':
        args.num_layers = 12
        args.num_heads = 12
        args.hidden_dim = 768
        args.mlp_dim = 3072
    elif args.model_name=='ViT-large':
        args.num_layers = 24
        args.num_heads = 16
        args.hidden_dim = 1024
        args.mlp_dim = 4096
    elif args.model_name=='ViT-huge':
        args.num_layers = 32
        args.num_heads = 16
        args.hidden_dim = 1280
        args.mlp_dim = 5120

    if args.mode=='train':
        model = VisionTransformer(image_size=args.crop_size, patch_size=(args.crop_size//args.patch),
                                  num_layers=args.num_layers, num_heads=args.num_heads,
                                  hidden_dim=args.hidden_dim, mlp_dim=args.mlp_dim, dropout=args.dropout,
                                  attention_dropout=0.0, num_classes=args.num_classes)

        # weights=ViT_B_16_Weights.IMAGENET1K_V1
        # model2 = vit_b_16(weights=weights)
        # for name, param in model2.named_parameters():
        #     # if name=='conv_proj.weight':
        #     #     param = torch.rand(768, 3, args.crop_size//args.patch, args.crop_size//args.patch)
        #     if name=='encoder.pos_embedding':
        #         param = torch.rand(1, args.patch*args.patch+1, 768)
        #     if name=='heads.head.weight':
        #         param = torch.rand(args.num_classes, 768)
        #     if name=='heads.head.bias':
        #         param = torch.rand(args.num_classes)
        #     for name2, param2 in model.named_parameters():
        #         if name2==name:
        #             param2=param
        #             # param2.requires_grad = False
        #             # if name=='conv_proj.weight' or name=='encoder.pos_embedding' or name=='heads.head.weight' or name=='heads.head.bias':
        #             #     param2.requires_grad = True

        # model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))


        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.batch, T_mult=1, eta_min=(args.lr*0.01))
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warmup, after_scheduler=base_scheduler)
        if args.mixup:
            mixup = MixUp(alpha=args.alpha)
        else:
            mixup = None
        model = model.to(args.device)

        for epoch in range(args.epoch):
            train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=args.device,
                            total_img_num=total_img_num, batch_size=args.batch, epoch=epoch, mixup=mixup)
            scheduler.step()
            val_one_epoch(model=model, loader=val_loader, device=args.device, criterion=criterion, epoch=epoch)
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        model = model.to(args.device)
        val_one_epoch(model=model, loader=val_loader, device=args.device, criterion=criterion)
    # print('epoch:',args.epoch,'num_ops:',args.num_ops,'magnitude:',args.magnitude,'alpha:',args.alpha)

def train_one_epoch(model, loader, optimizer, criterion, device, total_img_num, batch_size, epoch, mixup):
    model.train()
    acc = 0
    loss_cnt = 0
    if args.dataset=='resisc45':
        if args.randaugment:
            randaugment= transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude)
        for batch_idx, (fig) in enumerate(loader):
            # if args.randaugment:
            #     images = fig['image']
            #     images = randaugment(images)
            #     images = images.type(torch.float32)
            # else:
            images = fig['image'].type(torch.float32)
            labels = fig['label'].type(torch.int64)
            if mixup!=None:
                if np.random.rand() <= 0.8:
                    images, labels, rand_label, lambda_ = mixup((images, labels))
                else:
                    images, labels, rand_label, lambda_ = images, labels, torch.zeros_like(labels), 1.
                rand_label = rand_label.to(device)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if mixup!=None:
                loss = criterion(outputs, labels)*lambda_ + criterion(outputs, rand_label)*(1.-lambda_)
            else:
                loss = criterion(outputs, labels)
            loss_cnt += loss
            acc += torch.eq(outputs.argmax(-1), labels).float().mean().tolist()*100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\r[',epoch,']progress:',round(((batch_idx+1)*batch_size)/total_img_num*100,1),'%   ','train accuracy:',round(acc/(batch_idx+1),2),'%   ','train Loss:', round(loss_cnt.tolist()/(batch_idx+1), 3),end="")
        total_acc = acc/(batch_idx+1)
        total_loss = loss_cnt.tolist()/(batch_idx+1)
        # print('\r[',epoch,']train accuracy:',round(total_acc,2),'%   ','train Loss:', round(total_loss, 3),end="")
        wandb.log({
            "epoch": epoch,
            "acc": total_acc,
            "loss": total_loss
        })
    else:
        for batch_idx, (images, labels) in enumerate(loader):
            if mixup!=None:
                if np.random.rand() <= 0.8:
                    images, labels, rand_label, lambda_ = mixup((images, labels))
                else:
                    images, labels, rand_label, lambda_ = images, labels, torch.zeros_like(labels), 1.
                rand_label = rand_label.to(device)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if mixup!=None:
                loss = criterion(outputs, labels)*lambda_ + criterion(outputs, rand_label)*(1.-lambda_)
            else:
                loss = criterion(outputs, labels)
            loss_cnt += loss
            acc += torch.eq(outputs.argmax(-1), labels).float().mean().tolist()*100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\r[',epoch,']progress:',round(((batch_idx+1)*batch_size)/total_img_num*100,1),'%   ','train accuracy:',round(acc/(batch_idx+1),2),'%   ','train Loss:', round(loss_cnt.tolist()/(batch_idx+1), 3),end="")
        print('\r[',epoch,']train accuracy:',round(acc/(batch_idx+1),2),'%   ','train Loss:', round(loss_cnt.tolist()/(batch_idx+1), 3),end="")


@torch.no_grad()
def val_one_epoch(model, loader, device, criterion, epoch):
    model.eval()
    acc = 0
    loss_cnt = 0
    if args.dataset=='resisc45':
        for batch_idx, (fig) in enumerate(loader):
            images = fig['image'].type(torch.float32)
            labels = fig['label'].type(torch.int64)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(outputs.argmax(-1))
            # print(labels)
            acc += torch.eq(outputs.argmax(-1), labels).float().mean().tolist()*100
            loss_cnt += loss.tolist()
        total_acc = acc/(batch_idx+1)
        total_loss = loss_cnt/(batch_idx+1)
        # print('   val accuracy:',round(total_acc,2),'%   ','val Loss:',round(total_loss,3))
        wandb.log({
            "epoch": epoch,
            "val_acc": total_acc,
            "val_loss": total_loss
        })
    else:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(outputs.argmax(-1))
            # print(labels)
            acc += torch.eq(outputs.argmax(-1), labels).float().mean().tolist()*100
            loss_cnt += loss.tolist()
        print('   val accuracy:',round(acc/(batch_idx+1),2),'%   ','val Loss:',round(loss_cnt/(batch_idx+1),3))


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) 


def get_transform(args):
    train_transform = []
    test_transform = []
    if args.dataset == 'imagenet' or args.dataset=='imagenet-small':
        train_transform += [
            transforms.Resize((args.img_size, args.img_size))
        ]
        train_transform += [
            transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
        test_transform += [
            transforms.Resize((args.crop_size, args.crop_size))
        ]
        # test_transform += [
        #     transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=True)
        # ]
    elif args.dataset == 'stanfordcars':
        train_transform += [
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=True)
            transforms.Resize((args.img_size, args.img_size))
        ]
        test_transform += [
            transforms.Resize((args.crop_size, args.crop_size))
        ]
        if args.randaugment:
            train_transform += [
            transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude)
        ]
    elif args.dataset == 'food101':
        train_transform += [
            # transforms.RandomCrop(size=args.crop_size, padding=2, pad_if_needed=False)
            transforms.Resize((args.img_size, args.img_size))
        ]
        test_transform += [
            # transforms.RandomCrop(size=args.crop_size, padding=2, pad_if_needed=False)
            transforms.Resize((args.crop_size, args.crop_size))
        ]
    elif args.dataset == 'flowers-102':
        train_transform += [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
        test_transform += [
            transforms.Resize((args.crop_size, args.crop_size)),
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
    elif args.dataset == 'oxford-iiit-pet':
        train_transform += [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
        test_transform += [
            transforms.Resize((args.crop_size, args.crop_size)),
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
    elif args.dataset == 'caltech101':
        train_transform += [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)
        ]
    
    train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.dataset=='c100':
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform.append(CIFAR10Policy())
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        test_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    elif args.dataset=='imagenet' or args.dataset=='imagenet-small' or args.dataset=='food101' or args.dataset=='flowers-102' or args.dataset=='caltech101':
        mean, std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform.append(ImageNetPolicy())
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        test_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    elif args.dataset=='oxford-iiit-pet':
        mean, std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform.append(ImageNetPolicy())
        if args.randaugment:
            train_transform += [
                transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude)
            ]
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        test_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        print(f"No AutoAugment for {args.dataset}")
        train_transform += [
            transforms.ToTensor()
        ]
        test_transform += [
            transforms.ToTensor()
        ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class MixUp(object):
  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def __call__(self, batch):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x, y = batch
    lam = np.random.beta(self.alpha, self.alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def RandAugment_(item: dict):
    randaugment = transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude)
    item['image'] = randaugment(item['image'])
    return item

if __name__ == "__main__":
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        main()