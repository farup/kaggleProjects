import torch 
import numpy as np


from predict_stock import infer, load_model



model = load_model('model_final/Stock_state_dict_10.pt', input_size=10)
input = np.array([3.19, 3.24, 3.15])

model.eval()
prediction = infer(model, input=input)

print(f' The input sequence: {input}\n Prediction is {prediction.item():.2f} <=============================')




