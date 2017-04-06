from collections import Counter
from mpi4py import MPI
import json
import numpy as np
import time

area = {}
with open('melbGrid.json') as grid_data:
    data = json.load(grid_data)

    for line in data['features']:
        section = line['properties']['id']
        xmin = line['properties']['xmin']
        xmax = line['properties']['xmax']
        ymin = line['properties']['ymin']
        ymax = line['properties']['ymax']
        area[section] = [xmin, xmax, ymin, ymax]

#reads the twitter file and pre-preocess, store all location coordinates in a list
def readtwitter():
    print('This is master, rank = 0. Start reading twitter file...')
    start = time.time()
    twitter_list = []
    with open('smallTwitter.json') as tweet_data:
        for line in tweet_data:
            if line[0] == '{':
                line = line[:-1]
                if line[-1] == ',':
                    twitter_list.append(json.loads(line[:-1])['json']['coordinates']['coordinates'])
                else:
                    twitter_list.append(json.loads(line)['json']['coordinates']['coordinates'])
    end = time.time()
    print('Finish reading twitter file... time used:' + str(end - start))
    return twitter_list

#checkrow checkcolumn
def returnrow(ycoordinate):

    if ycoordinate >= area['A1'][2] and ycoordinate < area['A1'][3]:
        rownumber = 'A'
    elif ycoordinate >= area['B1'][2] and ycoordinate < area['B1'][3]:
        rownumber = 'B'
    elif ycoordinate >= area['C1'][2] and ycoordinate < area['C1'][3]:
        rownumber = 'C'
    elif ycoordinate >= area['D3'][2] and ycoordinate < area['D3'][3]:
        rownumber = 'D'
    else:
        rownumber = 'outofbound'
    return rownumber

def returncolumn(xcoordinate):
    if xcoordinate >= area['A1'][0] and xcoordinate < area['A1'][1]:
        columnnumeber = '1'
    elif xcoordinate >= area['A2'][0] and xcoordinate < area['A2'][1]:
        columnnumeber = '2'
    elif xcoordinate >= area['A3'][0] and xcoordinate < area['A3'][1]:
        columnnumeber = '3'
    elif xcoordinate >= area['A4'][0] and xcoordinate < area['A4'][1]:
        columnnumeber = '4'
    elif xcoordinate >= area['C5'][0] and xcoordinate < area['C5'][1]:
        columnnumeber = '5'
    else:
        columnnumeber = 'outofbound'
    return columnnumeber

#classify the coordinates as location IDs
def counter(locations):
    result = []
    for location in locations:
        if returnrow(location[1]) != 'outofbound' and returncolumn(location[0]) != 'outofbound':
            index = returnrow(location[1]) + returncolumn(location[0])
            #exclude outliers
            if index not in ['D1','D2','A5','B5']:
                result.append(index)
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    #if this is master,
    if rank == 0:
        locations = readtwitter()
        #partition the dataset so it can be scatter to all workers
        data = np.array_split(locations, size)
    else:
        data = None

    data = comm.scatter(data, root=0)
    local_result = counter(data)
    results = comm.allgather(local_result)

    #print final output on the master node
    if rank == 0:
        combined = [item for sublist in results for item in sublist]
        final_result = dict(Counter(combined))

        print('Order of Grid boxes based on the total number of tweets made in each box:')
        for w in sorted(final_result, key=final_result.get, reverse=True):
            print(str(w) + ': ' + str(final_result[w]) + ' tweets')
        print('\n')
        print('Order of the rows based on the total number of tweets in each row:')
        row_counter = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        for rownumber, rowcount in row_counter.items():
            for key, value in final_result.items():
                if str(rownumber) in str(key):
                    row_counter[rownumber] = row_counter[rownumber] + value

        for w in sorted(row_counter, key=row_counter.get, reverse=True):
            print(str(w) + ': ' + str(row_counter[w]) + ' tweets')
        print('\n')
        print('Order of the columns based on the total number of tweets in each column:')
        col_counter = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

        for colnumber, colcount in col_counter.items():
            for key, value in final_result.items():
                if str(colnumber) in str(key):
                    col_counter[colnumber] = col_counter[colnumber] + value

        for w in sorted(col_counter, key=col_counter.get, reverse=True):
            print(str(w) + ': ' + str(col_counter[w]) + ' tweets')

if __name__ == '__main__':
    main()


