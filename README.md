# SimCardinalityEstimator
This is an example code for dataset Fashion-Mnist
## Code Structure
```
.
./prepare_training_data.py
./simcard_evaluation.py
./train_global_models.py
./train_local_models.py
```
## Usage
- `python prepare_training_data.py`: Generate queries from given dataset and obtain the true cardinalities for each cluster.  
- `python train_global_models.py`: Train Global Model.  
- `python train_local_models.py`: Train All the Local Models.
- `python simcard_evaluation.py`: Evaluate cardinalities by using trained models.
## Trained Model and Generated Files From Code
You can download generated training queries and pretrained models from https://drive.google.com/file/d/12KS5nQjAneL7MCPK905J8XN8rCA9OstB/view?usp=sharing.
