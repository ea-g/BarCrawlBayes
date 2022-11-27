import pyro
import pyro.distributions as dist
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
    config.epochs = 1
    config.posterior_samples = 1000
    # params to use
    # 1) model load path
    # 2) model training config path
    # 3) batch size for bnn

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

    guide = partial(tyxe.guides.AutoNormal, init_scale=0.001)
    bnn = tyxe.VariationalBNN(model, prior, likelihood, guide)
    pyro.clear_param_store()
    optim = pyro.optim.AdamW({"lr": 1e-3, "weight_decay": 1e-4})
    elbos = []

    def callback(bnn, i, e):
        elbos.append(e)
        wandb.log({"ELBO": e})

    wandb.watch(model)
    with tyxe.poutine.local_reparameterization():
        bnn.fit(trainloader, optim, config.epochs, callback, device=device)

    pyro.get_param_store().save("param_store.pt")
    torch.save(bnn.state_dict(), "state_dict.pt")

    test_predictions = torch.cat([bnn.predict(x.to(device), num_predictions=config.posterior_samples, aggregate=True)
                                  for x, _ in iter(testloader)])
    test_gt = torch.cat([y for _, y in iter(testloader)])
    print(f"AUC ROC: {roc_auc_score(test_gt.detach().reshape(-1).cpu().numpy(), torch.sigmoid(test_predictions).detach().reshape(-1).cpu().numpy())}")
    print(f"Accuracy: {(test_gt.detach().reshape(-1).cpu().numpy()==torch.sigmoid(test_predictions).detach().reshape(-1).cpu().numpy().round()).mean()}")
    torch.save(test_predictions.detach().cpu(), "test_predictions.pt")
    wandb.save("param_stor.pt")
    wandb.save("state_dict.pt")
    wandb.save("test_predictions.pt")


if __name__ == "__main__":
    main()
