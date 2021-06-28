import os
import re
import random

# NOTE: Script is unordered due to usage when making it

# Create labels directories
#for i in range(1,16):
#    os.mkdir(f'data/chinese_mnist/{i}')
#
## Move every file
#for file in dirs:
#    EXTRACT_LABEL = "input_[0-9]+_[0-9]+_([0-9]+).jpg"
#    label = re.match(EXTRACT_LABEL, file).group(1)
#    os.rename(f'data/chinese_mnist/8/{file}',f'data/chinese_mnist/{label}/{file}')
#

for i in range(1,16):
    os.mkdir(f'data/chinese_mnist/val_images/{i}')


for i in range(1,16):
    files = os.listdir(f'data/chinese_mnist/train_images/{i}')
    random.shuffle(files)
    idx = len(files) // 10 * 3
    val_set = files[:idx]
    # train_set = files[idx:]

    for file in val_set:
        os.rename(f'data/chinese_mnist/train_images/{i}/{file}',f'data/chinese_mnist/val_images/{i}/{file}')