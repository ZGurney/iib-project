# Daily Note

23 October 2021

To-Do
- [x] Learn how to save models to produce plots later
- [x] Clamp regression standard deviation to prevent numerical issue
- [x] Change dual ConvCNP code to train on single datasets
- [x] Train three models and produce plots and performance metrics

Notes
- Saving models at each epoch to a folder `_experiments/experiment/saved_models/model1.tar` including epoch, loss as a tuple with value and error, model, optimiser parameters
- MLflow
	- Tracking: logging parameters, metrics and artifacts
	- Found it too challenging to integrate with Azure workflow, so just used Azure logging
- Saving model
	- 1. Native PyTorch save -> didn't work
	- 2. [Some directory manipulation](https://docs.microsoft.com/en-gb/azure/machine-learning/how-to-log-view-metrics) -> worked!
- How to load model on CPU?
	- `checkpoint = torch.load('models/model-1/model1.tar', map_location=torch.device('cpu'))` doesn't work
	- Try with joblib on Google Colab `size mismatch for conv.after_turn_layers.3.weight: copying a param with shape torch.Size([64, 16, 5]) from checkpoint, the shape in current model is torch.Size([32, 16, 5])` in `model.load_state_dict(…)`
- Numerical issue not a problem
- `orange_lamp` is first successful run of ordinary ConvCNP on classification
- `polite_stamp`is first successful run of ordinary ConvCNP on regression
- `musing_rose`is using Matern 5/2 kernel


Questions for Rich

1. Uncertainties for the classification data
2. How to compare 
3. NLL over epochs
4. High errors