# pytorch_utils
UtilityScripts and Functions for Pytorch 

engine.py contains complete training loops, kfold training loops and two staged staged kfold trianing loops

learning_rate_optimization is the automated version of learning rate tuning described by Data Scientist & Researcher Chris Deotte
Reference: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/488083

In order for the if __name__ == "__main__" part to work you will have to declare your own model and data pipeline in the script, the purpose of this part is only to demonstrate how you could use my functions
