import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def preprocess(img:np.array) -> np.array:
    trans = A.Compose(
        [A.Resize(75, 75),
        A.Normalize(),
        ToTensorV2()]
    )
    result = trans(image=img)
    return result['image']