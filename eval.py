import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, precision_score, recall_score, RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from torch.utils.data import DataLoader
from torchmetrics.functional import calibration_error
from torchvision.transforms import Lambda
from trainbayes import load_model
from utils.data_utils import *


def get_preds(config,
              model: nn.Module,
              device,
              test_loader: DataLoader
              ):
    """
    Gets the pseudo-probabilities from a Frequentist NN

    :param config:
        wandb config object
    :param model:
        trained pytorch model
    :param device:
        device to use, i.e. cpu or gpu from torch.device('your_device')
    :param test_loader:
        torch DataLoader containing test data
    :return:
        y_true, y_pred as numpy arrays
    """
    model.eval()
    gt_full = []
    pred_full = []

    for val_x, val_y in test_loader:
        with torch.no_grad():
            val_x, val_y = val_x.to(device), val_y.to(device)
            vpred = model(val_x)
            if config.num_classes < 2:
                vpred_proba = torch.sigmoid(vpred)
            else:
                vpred_proba = nn.functional.softmax(vpred, dim=-1)
                vpred_proba = vpred_proba[:, 1]
            gt_full.append(val_y.cpu().numpy().flatten())
            pred_full.append(vpred_proba.reshape(-1).cpu().numpy())

    gt_full = np.concatenate(gt_full)
    pred_full = np.concatenate(pred_full)

    return gt_full, pred_full


def add_scores(y_true, y_pred, score_dict: dict, decimals: int = 3):
    """
    Adds AUC, precision, recall, accuracy, and expected calibration scores to the scoring dictionary

    :param y_true:
        ground truth labels
    :param y_pred:
        predicted probabilities
    :param score_dict:
        dictionary holding all model scores
    :param decimals:
        number of decimal places to round. Default=3
    :return:
        None, updates the scoring dictionary, calibration plot, and AUC plot
    """
    score_dict['AUC'].append(roc_auc_score(y_true, y_pred).round(decimals))
    score_dict['accuracy'].append((y_pred.round() == y_true).mean().round(decimals))
    score_dict['precision'].append(precision_score(y_true, y_pred.round()).round(decimals))
    score_dict['recall'].append(recall_score(y_true, y_pred.round()).round(decimals))
    score_dict['expected calibration error'].append(calibration_error(torch.tensor(y_pred),
                                                                      torch.tensor(y_true),
                                                                      task='binary',
                                                                      n_bins=10).numpy().round(decimals))


def score_all():
    scores = {'model': [], 'AUC': [], 'accuracy': [], 'precision': [], 'recall': [], 'expected calibration error': []}
    auc_fig, auc_ax = plt.subplots()
    cal_fig, cal_ax = plt.subplots()
    ##########################################
    #              Frequentist NNs
    ##########################################
    # load up the configs and dataset
    config_paths = ['./configs/model_configs/res18best.yaml', './configs/model_configs/res34best.yaml']
    model_paths = ['./model_saves/res18best.h5', './model_saves/res34best.h5']
    fnns = ['Frequentist ResNet18', 'Frequentist ResNet34']
    scores['model'] += fnns
    load_dotenv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the pretrained model config
    for i in range(len(config_paths)):
        with open(config_paths[i]) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=os.getenv('WBKEY'))
        wandb.init(entity='ea-g', project='BarCrawlBayes', mode='disabled', config=config_paths[i])
        config = wandb.config

        # load the data and limit values based on clamp config
        testing = BarCrawlData('./data/test', './data/y_data_full.csv',
                               transforms=Lambda(lambda x: torch.clamp(x, min=-config.clamp_val, max=config.clamp_val)),
                               target_dtype=torch.long if config.num_classes > 1 else torch.float32)

        datatest = DataLoader(testing, batch_size=config.test_batch_size, shuffle=False)

        # load the model
        model = load_model(model_config, model_paths[i], device)

        # get preds and scores
        y_true, y_pred = get_preds(config, model, device, datatest)
        add_scores(y_true, y_pred, scores)

        # add to plots
        RocCurveDisplay.from_predictions(y_true, y_pred, name=fnns[i], ax=auc_ax)
        CalibrationDisplay.from_predictions(y_true, y_pred, n_bins=10, name=fnns[i], ax=cal_ax)

    ##################################################
    #             Bayesian Neural Networks
    ##################################################
    bnns = ['Bayesian ResNet18', 'Bayesian ResNet34']
    bnn_preds = ['./posterior_preds/BNNResNet18_pred.npy', './posterior_preds/BNNResNet34_pred.npy']
    y_true = np.load('./posterior_preds/test_gt.npy')
    scores['model'] += bnns

    for i in range(len(bnn_preds)):
        y_pred = np.load(bnn_preds[i])
        y_pred_mean = y_pred.mean(axis=0).flatten()
        add_scores(y_true, y_pred_mean, scores)

        # add to plots
        RocCurveDisplay.from_predictions(y_true, y_pred_mean, name=bnns[i], ax=auc_ax)
        CalibrationDisplay.from_predictions(y_true, y_pred_mean, n_bins=10, name=bnns[i], ax=cal_ax)


    auc_ax.set_title('ROC for FNNs and BNNs (mean)')
    auc_fig.show()
    cal_ax.set_title('Calibration Plots for FNNs and BNNs (mean)')
    cal_fig.show()
    score_df = pd.DataFrame(scores)
    return score_df


if __name__ == "__main__":
    score_data = score_all()