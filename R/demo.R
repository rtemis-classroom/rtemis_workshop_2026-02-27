# rtemis workshop
# 2026-02-26/27 E.D. Gennatas rtemis.org

# %% Packages ----
library(rtemis)
library(data.table)

# %% Data ----
dat <- read("./Data/data.xlsx")
inspect(dat)

# %% Check Data ----
check_data(dat)

# %% Preprocess Data ----
# Create a preprocessor object
prp <- preprocess(
  dat,
  config = setup_Preprocessor(character2factor = TRUE, remove_duplicates = TRUE)
)
# Get preprocessed data
datp <- preprocessed(prp)

check_data(datp)

# %% Models ----
# We train 4 models using different algorithms, but the same outer resampling folds.
# Please note that we are doing minimal tuning to reduce demo runtime.

## %% GLMNET ----
hospitalized48_glmnet <- train(
  datp,
  algorithm = "glmnet",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_glmnet"
)
plot_roc(
  hospitalized48_glmnet,
  main = "GLMNET",
  filename = "./Models/hospitalized48_glmnet_roc.svg"
)
plot_varimp(
  hospitalized48_glmnet,
  show_top = 11L,
  filename = "./Models/hospitalized48_glmnet_varimp.svg"
)

## %% CART ----
hospitalized48_cart <- train(
  datp,
  algorithm = "cart",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_cart"
)
plot_roc(
  hospitalized48_cart,
  main = "CART",
  filename = "./Models/hospitalized48_cart_roc.svg"
)
plot_varimp(
  hospitalized48_cart,
  filename = "./Models/hospitalized48_cart_varimp.svg"
)

## %% LightRF ----
hospitalized48_lightrf <- train(
  datp,
  algorithm = "lightrf",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_lightrf"
)
plot_roc(
  hospitalized48_lightrf,
  main = "LightRF",
  filename = "./Models/hospitalized48_lightrf_roc.svg"
)
plot_varimp(
  hospitalized48_lightrf,
  filename = "./Models/hospitalized48_lightrf_varimp.svg"
)

## %% LightGBM ----
hospitalized48_lightgbm <- train(
  datp,
  hyperparameters = setup_LightGBM(
    learning_rate = c(0.001, 0.01)
  ),
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_lightgbm"
)
plot_roc(
  hospitalized48_lightgbm,
  main = "LIghtGBM",
  filename = "./Models/hospitalized48_lightgbm_roc.svg"
)
plot_varimp(
  hospitalized48_lightgbm,
  filename = "./Models/hospitalized48_lightgbm_varimp.svg"
)

# %% Present Results ----
present(
  list(
    hospitalized48_glmnet,
    hospitalized48_cart,
    hospitalized48_lightrf,
    hospitalized48_lightgbm
  ),
  main = "Hospitalized at 48hrs",
  filename = "./Models/hospitalized48_boxplot.svg"
)
