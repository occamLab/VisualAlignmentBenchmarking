#!/bin/bash

IMAGE_IDX=0019

mkdir output_sfm

./build/sfm --image-1-path ~powerhorse/Documents/visualAlignmentComparison/1635895514.298703/cameraimage_0001.jpg --metadata-1-path ~powerhorse/Documents/visualAlignmentComparison/1635895514.298703/cameraimage_0001_metadata.json --image-2-path ~powerhorse/Documents/visualAlignmentComparison/1635895514.298703/cameraimage_${IMAGE_IDX}.jpg --metadata-2-path ~powerhorse/Documents/visualAlignmentComparison/1635895514.298703/cameraimage_${IMAGE_IDX}_metadata.json --vftype 4 --output-directory output_sfm --ratio-test 0.4

