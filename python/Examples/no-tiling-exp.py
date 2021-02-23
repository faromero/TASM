#!/usr/bin/env python
# coding: utf-8

# Non-tiled experiments

import os
import sys
import cv2
import glob
import tasm
import shutil
import pandas as pd
import numpy as np
from timeit import default_timer as now

resources_path = 'basics_resources'
if os.path.exists(resources_path):
  shutil.rmtree(resources_path)
os.mkdir(resources_path)

tasm.configure_environment({
  'default_db_path': os.path.join(resources_path, 'labels.db'),
  'catalog_path': os.path.join(resources_path, 'resources'),
})

def time_selection(video_name, metadata_id, label, first_frame=None, last_frame=None):
  start = now()
  selection = t.select(video_name, metadata_id, label, first_frame, last_frame) if first_frame is not None and last_frame is not None else t.select(video_name, metadata_id, label)
  num_objs = 0
  while True:
    obj = selection.next()
    if obj.is_empty():
      break
    num_objs += 1
  end = now()
  return (end - start) * 1e3, num_objs

# Initialize a tasm instance
t = tasm.TASM()

# Read in all videos and detections
video_dir = '/inputs/2017-12-14/'
detections_dir = '/inputs/metadata'

all_vid = glob.glob(os.path.join(video_dir, '*'))

# Label we're after
label = 'car'

processed_vid_inds = []
blacklist = [150]
i = 0
max_visit = 100
start = now()
for v in all_vid:
  bname = os.path.basename(v)
  ind = bname.split('.')[0]
  detection = '%s/%s.pkl' % (detections_dir, ind)

  if not os.path.exists(detection) or int(ind) in blacklist:
    continue

  print('== Storing %s without tiling ==' % bname, flush=True)
  processed_vid_inds.append(ind)

  detections = pd.read_pickle(detection)
  detections = detections.astype({'x1': int, 'y1': int, 'x2': int, 'y2': int})
  print(detections.head())

  # Add metadata about the video. 
  metadata_id = 'cars-%s' % ind
  metadata_info = []
  for _, r in detections.iterrows():
      if r.x1 < 0 or r.y1 < 0 or r.x2 < 0 or r.y2 < 0:
        print('Ignoring:', r)
        continue
      md = tasm.MetadataInfo(metadata_id, r.label, r.frame, r.x1, r.y1, r.x2, r.y2)
      metadata_info.append(md)

  t.add_bulk_metadata(metadata_info)

  # Store the video untiled
  untiled_video_name = 'no-tiled-%s' % ind
  t.store(v, untiled_video_name)

  i += 1
  if i >= max_visit:
    break

end = now()
e2e_store = (end - start) * 1e3

print('== Done storing videos ==')

# Retrieving objects of interest using TASM
tot_dur = []
tot_cars = 0
start = now()
for ind in processed_vid_inds:
  metadata_id = 'cars-%s' % ind
  untiled_video_name = 'no-tiled-%s' % ind
  print('== Selecting %s from %s ==' % (label, ind), flush=True)
  dur, num_cars = time_selection(untiled_video_name, metadata_id, label)
  tot_dur.append(dur)
  tot_cars += num_cars
end = now()
e2e_select = (end - start) * 1e3

print('Time to store videos and metadata: %.2f' % e2e_store)
print('Retrieved %d %s in %.2f' % (tot_cars, label, e2e_select), flush=True)
print('Total number of video chunks processed: %d' % len(processed_vid_inds))
print('Mean retrieval time %.2f, stddev: %.2f' % (np.mean(tot_dur), np.std(tot_dur)))

