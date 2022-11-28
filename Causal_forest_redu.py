import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
import time
import random
import shap
from scipy.special import softmax

# based on the code from: https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html

## Function receive a data frame, the name of the outcome (target), the most important feature (Treatment)
def causal_forest_redu(df,name_outcome,most_important_feature:str="",n_estimators:int=1000,optimize_parameters:bool=True,shap_values:bool=False,verbose:bool=False):
    tic = time.perf_counter()
    #train, test = train_test_split(df, test_size=0.2)
    feature_list = list(df.columns.values)
    
    if name_outcome in feature_list:
        feature_list.remove(name_outcome)        
    
    # If the treatment is not given, select a random feature 
    if most_important_feature == "":
        treatment = random.choice(feature_list)
    else:        
        treatment = most_important_feature
        
    if treatment != "target" and treatment != "Class" and treatment != "class":            
        feature_list.remove(treatment)
        
    covariates = feature_list
    Y = df[name_outcome] ## Outcome or target
    X = df[covariates] ## Features
    T = df[treatment] ## Info gain
    W = None # Counfounders            
    
    #Initialize
    #c_forest = CausalForestDML(criterion='het',n_estimators=n_estimators,cv=5,model_t=LassoCV(max_iter=10000), model_y=LassoCV(max_iter=10000))        
    c_forest = CausalForestDML(criterion='het',n_estimators=n_estimators,cv=5)  # default parameter is lasso
    
    if optimize_parameters == True:        
        c_forest.tune(Y, T, X=X, W=W)  # find the best parameters for the data
             
    c_forest.fit(Y, T, X=X, W=W)
    # estimate the CATE with the test set 
    #c_forest.const_marginal_ate(X_test)

    if verbose == True:        
        toc = time.perf_counter()
        print("\nTime elapsed to tune and fit Causal Forest: ",(toc-tic), "seconds")            
        
    heterogeneous_effect_feature_importance = c_forest.feature_importances_
    
    # sort from smallest contribution to largest contribution
    sort_index = np.argsort(heterogeneous_effect_feature_importance)    
    feature_importance = []
    for i in range(len(heterogeneous_effect_feature_importance)):
        feature_importance.append((heterogeneous_effect_feature_importance[sort_index[i]],feature_list[sort_index[i]]))
    
    # Include the treatment
    feature_importance.append((1,treatment))
    
    if shap_values == True:                  
        shap_values_c = c_forest.shap_values(X)                
        shap.summary_plot(shap_values_c[name_outcome][treatment].values,covariates)
        shap.summary_plot(shap_values_c[name_outcome][treatment].values,covariates,plot_type="bar")
        shap_values_c = shap_values_c[name_outcome][treatment].values
        
        # Calculate the feature importance (mean absolute shap value) for each feature
        importances = []    
        for i in range(np.shape(shap_values_c)[1]):
            importances.append(np.mean(np.abs(shap_values_c[:, i])))
                    
        # Calculate the normalized version
        importance_norm = softmax(importances)            
        shap_sorted = np.argsort(importance_norm)
        
        # return the names of the features (inices may be diferent because we remove the treatment)
        shap_features = []       
        for i in range(np.shape(shap_sorted)[0]):
            shap_features.append(feature_list[shap_sorted[i]])                        
        # Include the treatment
        shap_features.append(treatment)                 
        # If SHAP is requested, return the feature importance based in heterogeneity, and the feature importance based on SHAP
        return feature_importance, shap_features
    
    # If SHAP is not requested, return only the feature importance based on heterogeneity        
    return feature_importance
