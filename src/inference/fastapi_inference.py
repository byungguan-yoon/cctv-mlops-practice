import sys, os
from fastapi import FastAPI, File, Response
import uvicorn
from fastapi.responses import JSONResponse

import torch
import numpy as np

from cnn_model import FaceModel
from prepro import preprocess
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")

emp_feat = np.load('./emp_emb_feat.npy')
app = FastAPI()
model = FaceModel().eval()

@app.post("/inference/")
async def inference(image: bytes = File(...)):
    image = ImageEncoder.Decode(image, channels=3)
    image = preprocess(image)
    image = image.unsqueeze(dim=0)
    with torch.no_grad():
        img_feat = model(image)
    sim_score = np.dot(emp_feat, img_feat.detach().numpy().T)
    max_sim_score = sim_score.max()
    return {'max_sim_score' : float(max_sim_score)} # float() #  {'max_sim_score': }
#  float(max_sim_score)
if __name__ == '__main__':
    uvicorn.run("fastapi_inference:app", host="0.0.0.0", port=8786, reload=True)