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

## GLMNET ----
hospitalized48_glmnet <- train(
  datp,
  algorithm = "glmnet",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_glmnet"
)

## CART ----
hospitalized48_cart <- train(
  datp,
  algorithm = "cart",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_cart"
)

## LightRF ----
hospitalized48_lightrf <- train(
  datp,
  algorithm = "lightrf",
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_lightrf"
)

## LihtGBM ----
hospitalized48_lightgbm <- train(
  datp,
  hyperparameters = setup_LightGBM(
    learning_rate = c(0.001, 0.01)
  ),
  outer_resampling_config = setup_Resampler(seed = 650),
  outdir = "./Models/hospitalized48_lightgbm"
)

# %% Present Results ----
present(
  list(
    hospitalized48_glmnet,
    hospitalized48_cart,
    hospitalized48_lightrf,
    hospitalized48_lightgbm
  ),
  filename = "./Models/hospitalized48_boxplot.svg"
)
