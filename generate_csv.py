'''
This script will crawl through the output folder of VisualAlignment and create a dataframe
'''



from os import system
import sys
from glob import glob
import os
import pandas as pd
import json

def parse_path(path):
    norm_path = os.path.normpath(path)
    path_list = norm_path.split(os.sep)
    start_time_id = path_list[-4]
    image_id = path_list[-3]
    vf_id = path_list[-2]

    # get image path
    image_path = os.path.split(path)
    image_path = image_path[0] + "/matching_image.png"
    return start_time_id, image_id, vf_id, image_path


if __name__ == "__main__":
    # path = '/home/powerhorse/paul_tmp/VisualMatching/output/1635895978.43579/0002/1/results.json'
    path = '/home/powerhorse/paul_tmp/VisualMatching/output/1635895978.43579'


    column_names = ['start_time', 'picture_number', 'vf_type', 'is_valid', 
                    'numInliers', 'numMatches', 'residualAngle', 'tx', 'ty', 'tz', 
                    'yaw', 'depth', 'image_path', 'date', 'conditions', 'route_name']

    print(column_names)
    df = pd.DataFrame(columns = column_names)

    # iterate through all the results.jsons
    for json_path in glob(path + '/**/*results.json', recursive=True):

        start_time_id, image_id, vf_id, image_path = parse_path(json_path)

        # load in json and convert to a list
        f = open(json_path, 'r')
        res = json.load(f)

        # create row
        # TODO: make everything snake_case
        row = [start_time_id, image_id, vf_id] + list(res.values()) + ["NULL", image_path, "NULL", "NULL", "NULL"]

        # append row
        df.loc[len(df.index)] = row

    print(df)
    df.to_csv('11_3__1635895978_43579__ex1.csv')
