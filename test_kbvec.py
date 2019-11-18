import numpy as np
import time


print('loading WikiData.KB.50d...')
start = time.time()

kbvec = np.memmap('/home/lcw2/share/embeddings/WikiData.KB.50d/entity2vec.bin', dtype='float32', mode='r')
end = time.time()
print('loaded WikiData.KB.50d! Time cost:', (end - start), 'sec.')
print(type(kbvec))
print(kbvec[0:50])
print(kbvec[50:100])
print(kbvec.shape)
