from pathlib import Path
from PIL import Image
import numpy as np

root = Path('dataset')
for split, n in [('train', 20), ('val', 6), ('test', 6)]:
    for cls in ['tumor', 'no_tumor']:
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            if cls == 'tumor':
                arr[50:174, 50:174, 0] = np.clip(arr[50:174, 50:174, 0] + 100, 0, 255)
            else:
                arr[50:174, 50:174, 2] = np.clip(arr[50:174, 50:174, 2] + 100, 0, 255)
            Image.fromarray(arr).save(d / f'{cls}_{i}.png')
print('Synthetic dataset created: dataset/train, dataset/val, dataset/test with tumor/no_tumor images.')
