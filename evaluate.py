import joblib

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from omegaconf import OmegaConf



def evaluate(config): 
    print("Evaluating...")
    
    test_inputs = joblib.load(config.features.test_features_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)

    test_outputs = test_df["label"].values
    # get class name  
    class_names = test_df["sentiment"].unique().tolist() # holds positive and negative
    # class_names = ["postive", "negative"] # instead of this manually, get from df :)

    # load the trained model  
    model = joblib.load(config.train.model_save_path)

    # metrics 
    metric_name = config.evaluate.metric
    metric = {
        "accuracy": accuracy_score, 
        "f1_score": f1_score
    }[metric_name]            # specifies which metric to use --> [metric_name] in config file

    predicted_test_outputs = model.predict(test_inputs)
    
    result = metric(test_outputs, predicted_test_outputs)
    result_dict = {metric_name: float(result)}

    # save results using OmegaConf
    OmegaConf.save(result_dict, config.evaluate.results_save_path)



if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    evaluate(config)