# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import os
import cv2
import time
import torch
import base64
import asyncio
import numpy as np
import albumentations as A
from PIL import Image
from model import FaceModel
from albumentations.pytorch import ToTensorV2

from ts.torch_handler.base_handler import BaseHandler

def Encode(image, ext='jpg', quality=90):
    if ext == 'jpg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, image_serial = cv2.imencode('.jpg', image, encode_param)
    elif ext == 'png':
        _, image_serial = cv2.imencode('.png', image)
    image_bytes = image_serial.tobytes()
    return image_bytes

def Decode(image_bytes, channels=3):
    if channels==3:
        color = cv2.IMREAD_COLOR
    elif channels==1:
        color = cv2.IMREAD_GRAYSCALE

    image_serial = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_serial, color)
    return image

def album(img:np.array) -> np.array:
    trans = A.Compose(
        [A.Resize(75, 75),
        A.Normalize(),
        ToTensorV2()]
    )
    result = trans(image=img)
    return result['image']

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.model = None
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self._context = context

        if not torch.cuda.is_available() or properties.get("gpu_id") is None :
            raise RuntimeError("This model is not supported on CPU machines.")

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
        self.model = self._load_model(model_dir).to(self.device)
        self.emp_feat = np.load('emp_emb_feat.npy')
        self.initialized = True

    def _load_model(self, model_dir):
        model = FaceModel()
        model = self.model.load_from_checkpoint(os.path.join(model_dir, 'face_model.ckpt'))
        return model.eval()

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        bytes = data[0].get("data")
        if bytes is None:
            bytes = data[0].get("body")

        img_array = Decode(bytes)
        image = album(img_array)
        # d_img = base64.decodebytes(bytes)
        # img_array = np.frombuffer(d_img, dtype=np.float32).reshape((80,320,3))
        # image = np.resize(img_array, (320,320,3)) / 255.
        # image = torch.from_numpy(image).pin_memory()
        # image = torch.tensor(image, dtype=torch.float32).to(self.device, non_blocking=True)
        # image = image.permute(2,0,1).unsqueeze(0).to(device='cuda', non_blocking=True)
        return image.to(device='cuda')

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            model_output = self.model.forward(model_input.unsqueeze(0))
        return model_output

    async def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        sim_score = np.dot(self.emp_feat, inference_output.detach().numpy().T)
        max_sim_score = sim_score.max()
        # Take output from network and post-process to desired format
        # masks = inference_output.squeeze().detach().cpu()
        # masks = np.where(masks >= 0.9, 0, 1)
        # pixels = np.count_nonzero(masks, axis=(1,2))
        # pixels = np.count_nonzero(masks)
        # results = pixels < 2
        # results = results.tolist()
        return float(max_sim_score)

    async def main_async(self, input):
        result = await self.postprocess(input)
        return result

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        start = time.perf_counter()
        model_input = self.preprocess(data)
        print("preprocess time:",time.perf_counter()-start)
        s = time.perf_counter()
        model_output = self.inference(model_input)
        print("inference time:",time.perf_counter()-s)
        s = time.perf_counter()
        # return_value = self.postprocess(model_output)
        loop = asyncio.get_event_loop()
        return_value = loop.run_until_complete(self.main_async(model_output))
        # loop.close()
        print("postprocess time:",time.perf_counter()-s)
        print("total process time:",time.perf_counter()-start)

        return [return_value]