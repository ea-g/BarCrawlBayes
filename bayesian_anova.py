import numpy as np
import pymc as pm
import arviz as az


def load_data():
    """
    Loads in samples from the posterior predictive distribution of BNNResNet18 and BNNResNet34. Calculates the std. for
    each datapoint in the test set.
    :return:

    """
    bnn18 = np.load('./posterior_preds/BNNResNet18_pred.npy').std(axis=0).flatten()
    bnn34 = np.load('./posterior_preds/BNNResNet34_pred.npy').std(axis=0).flatten()
    return bnn18, bnn34


def anova():
    b18, b34 = load_data()
    with pm.Model() as m:
        mu0 = pm.Normal("mu0", mu=0, tau=0.0001)
        tau = pm.Gamma("tau", 0.001, 0.001)

        alpha2 = pm.Normal("alpha2", mu=0, tau=0.0001)
        # sum-to-zero constraint
        alpha1 = pm.Deterministic("alpha1", -(alpha2))

        mu_1 = mu0 + alpha1
        mu_2 = mu0 + alpha2

        pm.Normal("lik1", mu=mu_1, tau=tau, observed=b18)
        pm.Normal("lik2", mu=mu_2, tau=tau, observed=b34)

        onetwo = pm.Deterministic("alpha1-alpha2", alpha1 - alpha2)

        trace = pm.sample(5000)
    return trace


if __name__ == "__main__":
    t = anova()
    summary = az.summary(t, var_names=["alpha"], filter_vars="like", kind="stats")

