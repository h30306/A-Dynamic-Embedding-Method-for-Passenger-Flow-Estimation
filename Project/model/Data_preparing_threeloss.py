import pandas as pd
from tqdm import tqdm
import numpy as np
from itertools import combinations
from math import radians, sin, cos, asin, sqrt
import json
from IPython import embed

df2017 = pd.read_csv('../data/data_2017.csv')
df2018 = pd.read_csv('../data/data_2018.csv')
#df2019 = pd.read_csv('../data/data_2019.csv')
location = pd.read_csv('../data/location_info_newid.csv')

df2017 = df2017[df2017['id'] < 73]
df2018 = df2018[df2018['id'] < 73]

with open('../data/train_data_3loss.json', 'w') as f:
    f.write("")

def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return round(c * r, 2)

arrlocation = np.array(location)

alldf = []
for i in tqdm(range(len(arrlocation))):
    row = []
    targetlon, targetlat = arrlocation[i,1], arrlocation[i,2]
    for j in range(len(location)):
        row.append(haversine(targetlon, targetlat, arrlocation[j,1], arrlocation[j,2]))
    alldf.append(row)
corr = np.array(alldf)

df2017 = df2017.sort_values(['id', 'datetime'])
df2018 = df2018.sort_values(['id', 'datetime'])
#df2019 = df2019.sort_values(['id', 'datetime'])

df = pd.concat([df2017, df2018])
#df = pd.concat([df, df2019])
df.reset_index(drop=True, inplace=True)

bins = list(range(0,280000, 10))
bins.insert(0, -10000)
labels = ['v'+str(i) for i in list(range(1, len(list(range(0,280000, 10)))))]
labels.insert(0, 'v0')
df['binned'] = pd.cut(df['value'], bins=bins, labels=labels)
df.reset_index(drop=True, inplace=True)

df.duplicated(subset=['datetime', 'id']).sum()
df['id'] = df['id'].apply(lambda x : x+1)
darray = np.array(df)

d = {}
for i in tqdm(range(len(darray)-8)):
    if len(set(darray[i:i+8, 2])) == 1:
        k = str(darray[i:i+8, 0].tolist())
        if k in d:
            d[k].add(i)
        else:
            d[k] = {i}

res = {k: v for k, v in d.items() if len(v) > 1}

NSP_dict = {1:[2,13], 2:[1,66], 3:[4,50,51,66], 4:[3,5,60,61], 5:[4,6], 6:[5,7], 7:[6,8], 8:[7,9], 9:[8,10], 10:[9,11],
            11:[10,12], 12:[11], 13:[1,14], 14:[13,15], 15:[14,13], 16:[15,17], 17:[16,18], 18:[17,19], 19:[18,20], 20:[19,21],
            21:[20,22], 22:[21,23], 23:[22,57], 24:[25], 25:[24,26], 26:[25,27], 27:[26,28], 28:[27,29,65], 29:[28,30,31,65], 30:[29,68],
            31:[29,32], 32:[31,49,67,68], 33:[34,67], 34:[33,35,62,63], 35:[34,36], 36:[35,37], 37:[36,38], 38:[37,39], 39:[38,40], 40:[39,41],
            41:[40,42], 42:[41,43], 43:[42,44,45], 44:[43], 45:[43,46], 46:[45,47], 47:[46], 48:[68], 49:[32,50], 50:[3,49,65,69],
            51:[3,52], 52:[51,53], 53:[52,54], 54:[53,55], 55:[54,56], 56:[55,57], 57:[23,56], 58:[59], 59:[58,60], 60:[4,59],
            61:[4,65], 62:[34], 63:[34,64], 64:[63,69], 65:[28,29,50,61], 66:[2,3,69,71], 67:[32,33,69,70], 68:[30,48,67,70], 69:[50,64,66,67], 70:[67,68],
            71:[66,72], 72:[71,73], 73:[72]
            }

with open('../data/train_data_3loss.json', 'a') as f:
    for i,v in tqdm(res.items()):
        y_train_dist = []
        y_train_NSP = []
        X_train = []
        for j1, j2 in combinations(v, 2):
            train_data = darray[j1:j1+8,3].tolist()
            train_data.append('[SEP]')
            train_data.extend(darray[j2:j2+8,3].tolist())
            X_train.append(train_data)
            y_train_dist.append(corr[darray[j2,2], darray[j1,2]])
            if darray[j2,2] in NSP_dict[darray[j1,2]]:
                y_train_NSP.append(1)
            else:
                y_train_NSP.append(0)
        f.write(json.dumps({'y_dist': y_train_dist, 'y_NSP':y_train_NSP, 'x':X_train})+'\n')








