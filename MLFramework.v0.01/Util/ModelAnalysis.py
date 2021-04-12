from SALib.analyze import sobol
from SALib.sample import saltelli 


class ModelAnalysis:    
    @staticmethod
    def sensitivityAnalysis(model,mlKind,dfInputData,config):
        problem = {
            'num_vars': len(config._featureList),
            'names': config._featureList ,
            'bounds': []
        } 
        for xcol in config._featureList:
            problem['bounds'].append([dfInputData[xcol].min()-0.01, dfInputData[xcol].max()+0.01])  
        # Generate samples
        XSample = saltelli.sample(problem, 1000)
        # Run model (example)
        Y = model.predict(XSample)
        Y=Y.reshape((Y.shape[0],))  
        print('Sensitivity_'+mlKind,XSample.shape, Y.shape)
        Si = sobol.analyze(problem, Y, print_to_console=False)

        from pandas import DataFrame
        feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(config._featureList, Si['ST'])]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        df_feature_importances=DataFrame(feature_importances)
        df_feature_importances.columns = ['Feature','敏感度'+mlKind]
         

        return df_feature_importances
