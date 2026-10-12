# %%
import numpy as np
import pandas as pd
import pymc as pm

# %%
columns = [
    "rel_elev_50",
    "rel_elev_75 - rel_elev_25 AS rel_elev_iqr",
    "aspect_diff_50",
    "aspect_diff_75 - aspect_diff_25 AS aspect_diff_iqr",
    "length",
    "dist_to_water",
]

sql = f"SELECT {','.join(columns)}, class, case_id, annotator_id FROM lines"
df = pd.read_sql(sql, "sqlite:///model/annotated_lines.gpkg")


def normalize(x):
    """Normalize a series to have mean 0 and std 1."""
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std


df["rel_elev_50"], rel_elev_50_mean, rel_elev_50_std = normalize(df["rel_elev_50"])

df["rel_elev_iqr"] /= rel_elev_50_std  # Normalize by rel_elev_50 std

# Range will be [0, 18]. Note that preferable values is less than < 2, so this is okay.
df["aspect_diff_50"] /= 10
df["aspect_diff_iqr"] /= 10

df["length"] /= 1000  # Convert to km
df["dist_to_water"] /= 1000  # Convert to km

df["cid"], cid_to_case_id = pd.factorize(df["case_id"], sort=True)
df["uid"], uid_to_user_id = pd.factorize(df["annotator_id"], sort=True)


# %%
with pm.Model(
    coords={
        "obs": np.arange(len(df)),
        "case": cid_to_case_id.values,
        "user": uid_to_user_id.values,
        "user_m1": uid_to_user_id.values[1:],
        "real_class": [0, 1],
        "obs_class": [0, 1, 2],
    }
) as model:
    # Normalizing constants
    rel_elev_50_mean_ = pm.Data("rel_elev_50_mean", rel_elev_50_mean)
    rel_elev_50_std_ = pm.Data("rel_elev_50_std", rel_elev_50_std)

    # Priors for the coefficients
    chol, _, _ = pm.LKJCholeskyCov(
        "coef_chol_cov",
        n=8,
        eta=2,
        sd_dist=pm.HalfNormal.dist(sigma=0.2),
        compute_corr=True,
    )
    coef_raw = pm.Normal(
        "coef_raw", shape=(len(cid_to_case_id), chol.shape[0]), dims=("case", "coef")
    )
    coef_mu = pm.Normal(
        "coef_mu",
        mu=0,
        sigma=0.5,
        shape=chol.shape[0],
        dims=("coef"),
    )
    # rel_elev bounds
    coef = pm.math.dot(chol, coef_raw.T).T + coef_mu

    alpha = pm.Deterministic("a", coef[:, 0], dims=("case",))
    mu_rel_elev = pm.Deterministic("mu_rel_elev", coef[:, 1], dims=("case",))
    b_rel_elev_50 = pm.Deterministic("b_rel_elev_50", coef[:, 2], dims=("case",))
    b_aspect_diff_50 = pm.Deterministic("b_aspect_diff_50", coef[:, 3], dims=("case",))
    b_length = pm.Deterministic("b_length", coef[:, 4], dims=("case",))
    b_dist_to_water = pm.Deterministic("b_dist_to_water", coef[:, 5], dims=("case",))
    b_rel_elev_iqr = pm.Deterministic("b_rel_elev_iqr", coef[:, 6], dims=("case",))
    b_aspect_diff_iqr = pm.Deterministic(
        "b_aspect_diff_iqr", coef[:, 7], dims=("case",)
    )

    # A stronger prior is needed for the first user to avoid identifiability issues
    gamma_u1_raw = pm.TruncatedNormal(
        "gamma_u1_raw", mu=-3, sigma=1, upper=-2, shape=4, dims=("gamma_coef")
    )
    gamma_ui_raw_mu = pm.Normal(
        "gamma_ui_raw_mu", mu=-3, sigma=0.5, shape=4, dims=("gamma_coef")
    )

    gamma_ui_raw_chol, _, _ = pm.LKJCholeskyCov(
        "gamma_ui_raw_chol_cov",
        n=4,
        eta=2,
        sd_dist=pm.HalfNormal.dist(sigma=0.5),
        compute_corr=True,
    )
    gamma_ui_raw_raw = pm.Normal(
        "gamma_ui_raw_raw",
        shape=(len(uid_to_user_id) - 1, 4),
        dims=("user_m1", "gamma_coef"),
    )
    gamma_ui_raw = (
        pm.math.dot(gamma_ui_raw_chol, gamma_ui_raw_raw.T).T + gamma_ui_raw_mu
    )  # Shape: (n_users - 1, 4)

    gamma_raw = pm.math.concatenate(
        [gamma_u1_raw[None, :], gamma_ui_raw], axis=0
    )  # Shape (n_users, 4)
    gamma_raw = pm.math.stack(
        [
            pm.math.stack(
                [np.zeros(len(uid_to_user_id)), gamma_raw[:, 0], gamma_raw[:, 1]]
            ),
            pm.math.stack(
                [gamma_raw[:, 2], np.zeros(len(uid_to_user_id)), gamma_raw[:, 3]]
            ),
        ]
    )  # (2, 3, n_users)
    gamma = pm.math.softmax(gamma_raw, axis=1)
    gamma = pm.Deterministic("gamma", gamma, dims=("real_class", "obs_class", "user"))

    # Linear model
    cid = df["cid"].values
    uid = df["uid"].values

    mu = (
        alpha[cid]
        + b_rel_elev_50[cid] * (df["rel_elev_50"].values - mu_rel_elev[cid]) ** 2
        + b_aspect_diff_50[cid] * df["aspect_diff_50"]
        + b_length[cid] * (df["length"] - 1)
        + b_dist_to_water[cid] * df["dist_to_water"]
        + b_rel_elev_iqr[cid] * df["rel_elev_iqr"]
        + b_aspect_diff_iqr[cid] * df["aspect_diff_iqr"]
    )
    p = pm.Deterministic("p", pm.math.invlogit(mu), dims=("obs",))

    # Likelihood
    y_obs = pm.Categorical(
        "y_obs",
        p=(gamma[0][:, uid] * (1 - p) + gamma[1][:, uid] * p).T,
        observed=df["class"],
        dims=("obs",),
    )

# %%
with model:
    # Sample from the posterior
    idt = pm.sample(
        nuts_sampler="numpyro",
        idata_kwargs={
            "dims": {
                "coef_chol_cov_stds": ["coef"],
                "coef_chol_cov_corr": ["coef", "coef_bis"],
                "coef_chol_cov": ["coef_lt"],
                "gamma_ui_raw_chol_cov_stds": ["gamma_coef"],
                "gamma_ui_raw_chol_cov_corr": ["gamma_coef", "gamma_coef_bis"],
                "gamma_ui_raw_chol_cov": ["gamma_coef_lt"],
            },
            "log_likelihood": True,
        },
        target_accept=0.99,
        tune=2000,
    )

    idt.to_netcdf("model/model_params.nc")
