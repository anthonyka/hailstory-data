"""Pack the land mask into a small file for the daily job."""
import gzip, sys
import numpy as np

mask = np.load(sys.argv[1])
blob = gzip.compress(np.packbits(mask).tobytes(), 9, mtime=0)
with open(sys.argv[2], 'wb') as f:
    f.write(blob)
print('mask KB', len(blob) // 1024)
