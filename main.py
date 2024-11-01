import torch

import parameters as params
from model import SceneGenerator
from data_loader import load_image_dataset

torch.manual_seed(44)

train = load_image_dataset(root_dir="./data")

model = SceneGenerator()

model.train(train, params.epochs)

model.sample_images("cpu")

model.save_models("saved_models", epoch=params.epochs)
