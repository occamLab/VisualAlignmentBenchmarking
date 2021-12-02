#!/bin/bash

IMAGE_IDX=0003

eog /home/powerhorse/Documents/visualAlignmentComparison/1634152465.40384/${IMAGE_IDX}.jpg &

./build/visual_alignment --align-image-path ~/Documents/visualAlignmentComparison/1634152465.40384/alignimage_${IMAGE_IDX}.jpg --align-metadata-path ~/Documents/visualAlignmentComparison/1634152465.40384/alignimage_${IMAGE_IDX}_metadata.json --camera-image-path ~/Documents/visualAlignmentComparison/1634152465.40384/cameraimage_${IMAGE_IDX}.jpg --camera-metadata-path ~/Documents/visualAlignmentComparison/1634152465.40384/cameraimage_${IMAGE_IDX}_metadata.json --vftype 3 --output-directory output_sample --debug true

