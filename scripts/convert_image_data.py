#!/usr/bin/env python

import imageio
import numpy as np
import argparse
from os.path import isfile, join, sep, normpath, basename
from os import walk

parser = argparse.ArgumentParser(description='Image directory')
parser.add_argument('-d','--dir', help='Image directory',required=True)
parser.add_argument('-o','--out', help='Output base filename', required=True)
parser.add_argument('-l', '--label', help='Label type: pos = posivite, neg = negative', required=True)
parser.add_argument('-f','--fast', help='Skip checking size',required=False)
args = parser.parse_args()

check_size = (args.fast is None)

# validate command line input arguments
if not (args.label == "pos") and not (args.label == "neg"):
    raise ("Image has wrong shape: ", im.shape, " ", image_filename)

image_short_filenames = []
image_long_filenames = []
for (dirpath, dirnames, filenames) in walk(args.dir):
    for filename in filenames:
        if filename.endswith(".png"):
            if check_size:
                im = imageio.imread(dirpath + sep + filename)
                # skip if the shape is wrong
                if im.shape != (25, 100, 3): 
                    continue

            image_long_filenames.append(dirpath + sep + filename)
            image_short_filenames.append(filename.split(".")[0])
            
num_images = len(image_long_filenames)
print("Found %i images"%(num_images))
print(image_long_filenames[0])
print(image_short_filenames[0])

data_tensor = np.zeros(shape=(num_images, 25, 100, 3), dtype=float, )
label_tensor = None
ids_tensor = np.array(image_short_filenames, dtype=np.string_)

if args.label == "pos":
    print("Assigning 1's as positive labels")
    label_tensor = np.ones((num_images, 1), dtype=int)
elif args.label == "neg":
    print("Assigning 0's as negative labels")
    label_tensor = np.zeros((num_images, 1), dtype=int)
else:
  raise ("Wrong label type" + args.label)  


for image_index in range(num_images):
    image_filename = image_long_filenames[image_index]
    im = imageio.imread(image_filename)
    if im.shape == (25, 100, 3):
        data_tensor[image_index] = im
    if (image_index % 1000) == 0:
        print("Number of images processed: {}".format(image_index))
    

output_data_filename = args.out + "_data.npz"
output_labels_filename = args.out + "_labels.npz"
output_ids_filename = args.out + "_ids.npz"
np.savez_compressed(output_data_filename, data_tensor)
np.savez_compressed(output_labels_filename, label_tensor)
np.savez_compressed(output_ids_filename, ids_tensor)
