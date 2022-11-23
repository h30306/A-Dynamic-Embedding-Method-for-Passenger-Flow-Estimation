import pandas as pd 
import numpy as np 
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import argparse

def haversine(lon1, lat1, lon2, lat2): # 經度1，緯度1，經度2，緯度2

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r * 1000

def normalization(data):
    '''
    column you want to normalize
    '''
    max_distance = data.max()
    min_distance = data.min()
    return [(d-min_distance)/(max_distance-min_distance) for d in list(data)]

def weighted(data):
    return 1-data.values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_csv', type = str, default = '../data/mrt_vd.csv',
                        help = 'The original CSV file you want to convert')
    parser.add_argument('--output_dir', type = str, default = '../data/distance_matrix.txt',
                        help = 'The direction you want to save the output file')
    parser.add_argument('--normalization', type = 'store_true',
                        help = 'Whether to normalization')
    parser.add_argument('--weighted', type = 'store_true',
                        help = 'Whether to  weighted')
    args = parser.parse_args()

    original_data = pd.read_csv(args.original_csv)
    distance_list = []
    max_dist = 0
    for index1 in tqdm(list(original_data['Id'])):
        for index2 in list(original_data['Id']):
            if index1 == index2:
                distance_list.append(0)
            else:
                lon1 = original_data.loc[original_data['Id']==index1]['經度']
                lat1 = original_data.loc[original_data['Id']==index1]['緯度']
                lon2 = original_data.loc[original_data['Id']==index2]['經度']
                lat2 = original_data.loc[original_data['Id']==index2]['緯度']
                distance = haversine(lon1, lat1, lon2, lat2)
                distance_list.append(distance)
    node = list(original_data['Id'])
    node1 = node*len(node)
    node1.sort()
    node2 = node*len(node)
    distance_matrix = pd.DataFrame({'node1':node1, 'node2':node2, 'distance':distance_list})

    #normalization
    if args.normalization:
        distance_matrix['distance'] = normalization(distance_matrix['distance'])
    #weighted
    if args.weighted:
        distance_matrix['distance'] = weighted(distance_matrix['distance'])

    f = open(args.output_dir, 'w')
    for index, row in distance_matrix.iterrows(): 
        f.write("{} {} {}\n".format(int(row['node1']), int(row['node2']), row['distance']))
    f.close()

if __name__ == "__main__":
    main()