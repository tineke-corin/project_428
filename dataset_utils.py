from datasets import load_dataset, concatenate_datasets
from app.services.utils import Point, BoundingBox, is_within_box, intersection_over_union, point_from_normalised, box_from_normalised
from PIL import Image
import cv2
import numpy as np
import os

"""
  get_ground_truths(labels, image_width, image_height)

  Given a set of labels for an image, that are expressed in normalised
  coordinates, calculate the bounding box ground truths in (x, y) pixel
  coordinates.
"""
def get_ground_truths(labels, image_width, image_height):
  truths = []    

  for labeled_point in labels:
    # These are normalised coordinates: centre X, centre Y and region width and height,
    # expressed as a proportion of the total image size.
    c_x, c_y, width, height = labeled_point

    # Turn centre point & region size into top-left and bottom-right coordinates
    left_x = c_x - (width/2)
    top_y = c_y - (height/2)
    right_x = c_x + (width/2)
    bottom_y = c_y + (height/2)

    # Then turn those into pixel coordinates from the normalised values
    pt1 = point_from_normalised(left_x, top_y, image_width, image_height)
    pt2 = point_from_normalised(right_x, bottom_y, image_width, image_height)

    bx = BoundingBox(start_point=pt1, end_point=pt2)
    truths.append( bx )
  return truths


"""
    annotator(path, debug)

    Returns an annotation function that can be passed to dataset.map() to
    load annotation data from files into the dataset structure.
"""
def annotator(path, debug):
  def add_annotations(data_point):
    img_name = os.path.basename(data_point["image"].filename)
    annotation_path = os.path.join(path, img_name.replace(".png", ".txt"))
    data_point['file'] = img_name
    data_point['faces'] = []
    data_point['plates'] = [] 
    with open(annotation_path, "r") as f:
      content = f.read().split('\n')
      for row in content:
        values = [float(val) for val in row.split()]
        if len(values) == 0:
            continue
        label_class, left_x, top_y, right_x, bottom_y = values
        if (label_class == 0):
          data_point['faces'].append([ left_x, top_y, right_x, bottom_y ])
        elif(label_class == 1):
          data_point['plates'].append([ left_x, top_y, right_x, bottom_y ])
        elif(label_class == 2):
          if debug:
            print(f'{img_name} unidentifiable face, skipping')
        elif(label_class == 3):
          if debug:
            print(f'{img_name} unidentifiable LP, skipping')
    return data_point
  return add_annotations

"""
  load_test_dataset(debug, shuffle)
  
  Load the images to be used for testing, and for each one load its annotations
  and turn these into a dataset.
"""
def load_test_dataset(dataset_directory, debug=False, shuffle=True):
  datasets = {}
  for subset in [ "strasbourg", "stuttgart", "zurich" ]:
    # Load images from the dataset
    dir = f'dataset/images/{subset}'
    dataset = load_dataset("imagefolder", data_dir=f'{dataset_directory}/images/{subset}')
    annotation_path = f'{dataset_directory}/annotations/{subset}'

    # Map the annotations onto the images
    annotation_function = annotator(annotation_path, debug)
    dataset = dataset.map(annotation_function)
    datasets[subset] = dataset

  # For each dataset, pull out the training split, concatenate them all together then
  # shuffle the order
  dataset = concatenate_datasets( [ d['train'] for d in datasets.values() ])

  if shuffle:
    dataset = dataset.shuffle()
  return dataset
