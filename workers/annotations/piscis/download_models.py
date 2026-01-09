import numpy as np

from huggingface_hub import get_collection, HfFileSystem
from piscis import Piscis

Piscis(model_name='20230616')
Piscis(model_name='20230709')
Piscis(model_name='20230905')
Piscis(model_name='20251212')

fs = HfFileSystem()
collection_name = 'rajlab/raj-lab-piscis-models-6628d842730129abe061bef5'
collection = get_collection(collection_name)
for model in collection.items:
    model_path = model.item_id
    model_name = model_path.split('/')[-1]
    fs.download(f'{model_path}/{model_name}', f'/root/.piscis/models/{model_name}')
