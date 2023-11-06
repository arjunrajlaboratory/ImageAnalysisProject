import numpy as np

from piscis import Piscis

Piscis(model_name='20230616')
Piscis(model_name='20230709')
Piscis(model_name='20230905', batch_size=2).predict(np.zeros((256, 256)))
