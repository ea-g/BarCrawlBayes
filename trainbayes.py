import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag
import tyxe
import wandb
import yaml
from dotenv import load_dotenv
from functools import partial
from models.resnet1d import *
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Lambda
from utils.data_utils import *


# bernoulli version


def load_model(model_config: dict, model_state_path: str, device) -> nn.Module:
    model = ResNet1D(
        in_channels=3,
        base_filters=model_config['base_filters']['value'],
        kernel_size=model_config['kernel_size']['value'],
        stride=1,
        n_block=model_config['n_block']['value'],
        groups=1,
        n_classes=model_config['num_classes']['value'],
        downsample_gap=model_config['downsample_gap']['value'],
        increasefilter_gap=model_config['increasefilter_gap']['value'],
        drop_out=model_config['drop_out']['value'])
    model.to(device)
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    return model


def main():
    load_dotenv()
    wandb.login(key=os.getenv('WBKEY'))
    wandb.init(entity='ea-g', project='BarCrawlBayes')
    config = wandb.config  # Initialize config
    config.model_config_path = './configs/model_configs/res18best.yaml'
    config.batch_size = 128
    config.model_state_path = './model_saves/res18best.h5'
    config.epochs = 80
    config.posterior_samples = 64
    config.lr = 0.0005
    config.roc_auc_thresh = 0.845
    config.loc_from_pretrained = True
    config.eval_freq = 3
    config.flipout = True

    # load the pretrained model config
    with open(config.model_config_path) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # make the model
    model = load_model(model_config, config.model_state_path, device)

    # make the dataset and loader
    training = BarCrawlData('./data/train', './data/y_data_full.csv',
                            transforms=Lambda(lambda x: torch.clamp(x, min=-model_config['clamp_val']['value'],
                                                                    max=model_config['clamp_val']['value'])),
                            target_dtype=torch.long if model_config['num_classes']['value'] > 1 else torch.float32)

    trainloader = DataLoader(training, batch_size=config.batch_size, **kwargs)

    testing = BarCrawlData('./data/test', './data/y_data_full.csv',
                           transforms=Lambda(lambda x: torch.clamp(x, min=-model_config['clamp_val']['value'],
                                                                   max=model_config['clamp_val']['value'])),
                           target_dtype=torch.long if model_config['num_classes']['value'] > 1 else torch.float32)

    testloader = DataLoader(testing, batch_size=config.batch_size, **kwargs)

    # define the BNN
    prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)),
                                 expose_all=False, hide_module_types=(nn.BatchNorm1d,))
    likelihood = tyxe.likelihoods.Bernoulli(len(trainloader.sampler), event_dim=1)

    # a guide using mean field--pretrained weights used as location init
    if config.loc_from_pretrained:
        guide = partial(tyxe.guides.AutoNormal,
                        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(model),
                        init_scale=1e-2,
                        # max_guide_scale=0.01
                        )
    else:
        guide = partial(tyxe.guides.AutoNormal)
    bnn = tyxe.VariationalBNN(model, prior, likelihood, guide)
    pyro.clear_param_store()
    optim = pyro.optim.Adam({"lr": config.lr})

    best = {'auc_roc': config.roc_auc_thresh,
            'loss': 5}
    # callback for logging stats to wandb and evaluating every x epochs
    def callback(b, i, avg_elbo):
        # evaluating is expensive, do so every 3/user defined epochs
        if (not i % config.eval_freq) or (i == config.epochs-1):
            loss = torch.zeros(1).to(device)
            preds_full = []
            gt_full = []
            b.eval()
            for x, y in iter(testloader):
                with torch.no_grad():
                    logits = b.predict(x.to(device), num_predictions=config.posterior_samples, aggregate=False)
                    loss += nn.functional.binary_cross_entropy_with_logits(logits.mean(axis=0), y.to(device), reduction='sum')
                    preds_full.append(torch.sigmoid(logits).cpu().numpy())
                    gt_full.append(y.reshape(-1).cpu().numpy())

            gt_full = np.concatenate(gt_full)
            preds_full = np.concatenate(preds_full, axis=1)
            roc_auc = roc_auc_score(gt_full, preds_full.mean(axis=0).flatten())
            wandb.log({
                "Test Accuracy": (preds_full.mean(axis=0).flatten().round() == gt_full).mean(),
                "Test Loss": loss/len(gt_full),
                "Test AUC_ROC": roc_auc,
                "ELBO": avg_elbo
            })
            # save the best models based on validation loss or auc roc
            if (roc_auc > best['auc_roc']) or (loss/len(gt_full) < best['loss']):
                np.save(f'predictions_best.npy', preds_full)
                wandb.save(f'predictions_best.npy')
                torch.save(b.state_dict(), f"state_dict_best.pt")
                wandb.save(f"state_dict_best.pt")
                if roc_auc > best['auc_roc']:
                    best['auc_roc'] = roc_auc
                if loss/len(gt_full) < best['loss']:
                    best['loss'] = loss/len(gt_full)

            b.train()

    wandb.watch(model)

    # fit the model using Variational inference, saving the end state
    if config.flipout:
        context = tyxe.poutine.flipout
    else:
        context = tyxe.poutine.local_reparameterization
    with context():
        bnn.fit(trainloader, optim, config.epochs, callback, device=device)

    # save the final model
    pyro.get_param_store().save("param_store.pt")
    torch.save(bnn.state_dict(), "state_dict.pt")
    wandb.save("param_store.pt")
    wandb.save("state_dict.pt")


if __name__ == "__main__":
    main()
