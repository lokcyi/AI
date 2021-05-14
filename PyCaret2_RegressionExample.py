#%% check version
from pycaret.utils import version
version()

# %% 1. Loading Dataset
from pycaret.datasets import get_data
data = get_data('insurance')

# %% 2. Initialize Setup
from pycaret.regression import *
reg1 = setup(data, target = 'charges', session_id=123, log_experiment=True, experiment_name='insurance1')

#%% 3. Compare Baseline
best_model = compare_models(fold=5)

# %% 4. Create Model
lightgbm = create_model('lightgbm')

# %%
import numpy as np
lgbms = [create_model('lightgbm', learning_rate=i) for i in np.arange(0.1,1,0.1)]

# %%
print(len(lgbms))

# %% 5. Tune Hyperparameters
tuned_lightgbm = tune_model(lightgbm, n_iter=50, optimize = 'MAE')

# %%
tuned_lightgbm

# %% 6. Ensemble Model
dt = create_model('dt')

# %%
bagged_dt = ensemble_model(dt, n_estimators=50)

# %%
boosted_dt = ensemble_model(dt, method = 'Boosting')

# %% 7. Blend Models
blender = blend_models(estimator_list = [dt])

# %% 8. Stack Models
stacker = stack_models(estimator_list = compare_models(n_select=5, fold = 5, include = ['dt','rf','xgboost']))

# %% 9. Analyze Model
plot_model(dt)

# %%
plot_model(dt, plot = 'error')

#%%
plot_model(dt, plot = 'feature')

#%%
evaluate_model(dt)

#%% 10. Interpret Model
interpret_model(lightgbm)

#%%
interpret_model(lightgbm, plot = 'correlation')

#%%
interpret_model(lightgbm, plot = 'reason', observation = 12)

#%% 11. AutoML()
best = automl(optimize = 'MAE')
best

#%% 12. Predict Model
pred_holdouts = predict_model(lightgbm)
pred_holdouts.head()

#%%

new_data = data.copy()
new_data.drop(['charges'], axis=1, inplace=True)
predict_new = predict_model(best, data=new_data)
predict_new.head()

#%% 13. Save / Load Model
save_model(best, model_name='best-model')

#%%
loaded_bestmodel = load_model('best-model')
print(loaded_bestmodel)

#%%
from sklearn import set_config
set_config(display='diagram')
loaded_bestmodel[0]

#%%
from sklearn import set_config
set_config(display='text')

# %% 14. Deploy Model
deploy_model(best, model_name = 'best-aws', authentication = {'bucket' : 'pycaret-test'})

#%% 15. Get Config / Set Config
X_train = get_config('X_train')
X_train.head()

#%%
get_config('seed')

#%%
from pycaret.regression import set_config
set_config('seed', 999)

#%%
get_config('seed')

#%% 16. MLFlow UI 
# #!mlflow ui

#%% End