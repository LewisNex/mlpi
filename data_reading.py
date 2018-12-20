import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore 

i32 = type(1)
f64 = type(0.1)
stype = type("string")


columns =   ["event", "jet", "type", "eta", "phi", "pt", "E", "N", "Nq", "mass", "girth", "broad", "M1/2", "e", "eq", "pull_eta", "pull_phi", "sj1", "sj2", "pt<r/2", "pt<r", "pt<3r/2"]
col_types = [ i32,     i32,   stype,  f64,   f64,   f64,  f64, i32, i32,  f64,    f64,     f64,     f64,    f64, f64,  f64,        f64,        f64,   f64,   f64,      f64,    f64]
dtypes = {col:dtype for col, dtype in zip(columns, col_types)}
converters = {"type": lambda letter: 1 if letter == "W" else 0}


def standardize(X, demean=True, unitnorm=True, pca=False):
    # de-mean the data and rescale to unit norm, 
    rescaled = StandardScaler(with_mean=demean, with_std=unitnorm).fit_transform(X)
    if pca:
         return pd.DataFrame(PCA(n_components="mle").fit_transform(rescaled))
    else:
        return pd.DataFrame(rescaled, columns=X.columns)
    
def load_obs_data(paths):
    # Load in and preproc data from specified paths
    raw_data = pd.concat([pd.read_csv(path, header=None, names=columns, converters=converters, index_col=False).dropna()
            for path in paths])
    # Remove jets with too few particles or bad charged eccentricities
    raw_data = raw_data[(raw_data["N"] >= 4) & (raw_data["eq"] != 0) & (raw_data["pt<r"] > 0)]
    # Define some new vars
    raw_data["Nq/E"] = raw_data["Nq"] / raw_data["E"]
    raw_data["Emean"] = raw_data["E"] / raw_data["N"]
    raw_data["pull_mag2"] = raw_data["pull_eta"] * raw_data["pull_eta"] + raw_data["pull_phi"] * raw_data["pull_phi"]
    raw_data["sub_rat"] = raw_data["sj2"] / raw_data["sj1"]

    raw_data["log_e"] = np.log(raw_data["e"])
    raw_data["log_eq"] = np.log(raw_data["eq"])
    raw_data["log_pt<r"] = np.log(raw_data["pt<r"])
    raw_data["exp_pt<r"] = np.exp(raw_data["pt<r"])
    
    # Remove the values most prone to outliers
    outlier_cols = ["e", "eq"]
    return raw_data[np.abs(zscore(raw_data[outlier_cols]) < 3).all(axis=1)].sample(frac=1).reset_index(drop=True)
    
    