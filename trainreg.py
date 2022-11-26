import numpy as np
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.transforms import Lambda
from models.resnet1d import ResNet1D
from utils.data_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import wandb
import os
from dotenv import load_dotenv


def train_epoch(config,
                model: nn.Module,
                device,
                train_loader: DataLoader,
                optimizer,
                loss_func,
                step: int):
    model.train()
    prog = tqdm(train_loader, desc='Training', leave=False)
    train_loss = 0
    for idx, batch in enumerate(prog):
        x, y = tuple(t.to(device) for t in batch)
        pred = model(x)
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        train_loss += loss.item()

    wandb.log({
        "Train Loss": train_loss/len(train_loader)
    })


def eval_loop(config,
              model: nn.Module,
              device,
              test_loader: DataLoader,
              loss_func
              ):
    model.eval()
    test_loss = 0
    gt_full = []
    pred_full = []

    for val_x, val_y in test_loader:
        with torch.no_grad():
            val_x, val_y = val_x.to(device), val_y.to(device)
            vpred = model(val_x)
            test_loss += loss_func(vpred, val_y).item()
            if config.num_classes < 2:
                vpred_proba = torch.sigmoid(vpred)
            else:
                vpred_proba = nn.functional.softmax(vpred, dim=-1)
                vpred_proba = vpred_proba[:, 1]
            gt_full.append(val_y.cpu().numpy().flatten())
            pred_full.append(vpred_proba.reshape(-1).cpu().numpy())

    gt_full = np.concatenate(gt_full)
    pred_full = np.concatenate(pred_full)
    wandb.log({
        "Test Accuracy": (pred_full.round() == gt_full).mean(),
        "Test Loss": test_loss/len(test_loader),
        "Test AUC_ROC": roc_auc_score(gt_full, pred_full)
    })
    return test_loss/len(test_loader)


def main():
    load_dotenv()
    conf_path = './configs/res34_sweep.yaml'
    # comment out the next two lines to manually set params. also remove config part from `wandb.init()`
    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=os.getenv('WBKEY'))
    wandb.init(entity='ea-g', project='BarCrawlBayes', config=config)

    config = wandb.config  # Initialize config
    # config.batch_size = 256  # input batch size for training (default: 64)
    # config.test_batch_size = 256  # input batch size for testing (default: 1000)
    # config.epochs = 50  # number of epochs to train (default: 10)
    # config.lr = 1e-3  # learning rate
    # config.weight_decay = 1e-4
    # config.no_cuda = False  # disables CUDA training if True
    # config.drop_out = 0.5
    # config.base_filters = 64
    # config.kernel_size = 7
    # config.n_block = 16
    # config.downsample_gap = 2
    # config.increasefilter_gap = 4
    # config.clamp_val = 3
    # config.pos_weight = 1.5



    # define the device
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # parameters for defining ResNet depth
    # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
    # 34 layer (16*2+2): 16, 2, 4
    # 98 layer (48*2+2): 48, 6, 12
    model = ResNet1D(
        in_channels=3,
        base_filters=config.base_filters,
        kernel_size=config.kernel_size,
        stride=1,
        n_block=config.n_block,
        groups=1,
        n_classes=config.num_classes,
        downsample_gap=config.downsample_gap,
        increasefilter_gap=config.increasefilter_gap,
        drop_out=config.drop_out)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.num_classes > 1:
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2.]).to(device))
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight]).to(device))
    n_epoch = config.epochs
    step = 0

    # create the data sets
    training = BarCrawlData('./data/train', './data/y_data_full.csv',
                            transforms=Lambda(lambda x: torch.clamp(x, min=-config.clamp_val, max=config.clamp_val)),
                            target_dtype=torch.long if config.num_classes > 1 else torch.float32)
    testing = BarCrawlData('./data/test', './data/y_data_full.csv',
                           transforms=Lambda(lambda x: torch.clamp(x, min=-config.clamp_val, max=config.clamp_val)),
                           target_dtype=torch.long if config.num_classes > 1 else torch.float32)

    datatrain = DataLoader(training, batch_size=config.batch_size, shuffle=True, **kwargs)
    datatest = DataLoader(testing, batch_size=config.test_batch_size, shuffle=False, **kwargs)

    # turn on logging
    wandb.watch(model)

    best_val = 100
    for e in tqdm(range(n_epoch), desc='epoch', leave=False):
        train_epoch(config, model, device, datatrain, optimizer, loss_func, step)
        val_loss = eval_loop(config, model, device, datatest, loss_func)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best.h5')

    # save the model
    torch.save(model.state_dict(), "end.h5")
    wandb.save('end.h5')
    wandb.save('best.h5')


if __name__ == "__main__":
    main()
