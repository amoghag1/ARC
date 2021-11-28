#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
import copy
import itertools

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_25d8a9c8(x):
    ans = np.zeros(np.shape(x))
    for i in range(len(x)):
        if(np.all(x[i] == x[i][0])):
            ans[i].fill(5)
    return ans

def solve_e179c5f4(x):
    a = copy.deepcopy(x)
    j=0
    flag = 0
    nrow = a.shape[0]
    nclm = a.shape[1]
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            a[i][k] = 8

    for i in range(nrow):
        if (flag==0):
            a[nrow-1][j]=1
            nrow-=1
            j+=1
            if(j==nclm):
                j-=2
                flag=1
        elif (flag==1):
            a[nrow-1][j]=1
            nrow-=1
            j-=1
            if(j==-1):
                j=1
                flag=0

    return a

def solve_ea786f4a(x):
    b = copy.deepcopy(x)
    np.fill_diagonal(b, 0, wrap=True)
    np.fill_diagonal(np.fliplr(b), [0])
    return b

def solve_05269061(x):
    e = copy.deepcopy(x)
    (nrow, ncol) = np.shape(e)
    colors = []
    locations = []
    for i in range(ncol-1):
        if((len(colors))<=4):
            if(e[0][i] != 0 and e[0][i] not in colors):
                colors.append(e[0][i])
                locations.append((0,i))
            if(e[i][0] != 0 and e[i][0] not in colors):
                colors.append(e[i][0])
                locations.append((i,0))
            if(e[i][ncol-1] != 0 and e[i][ncol-1] not in colors):
                colors.append(e[i][ncol-1])
                locations.append((i,ncol-1))
            if(e[ncol-1][i] != 0 and e[ncol-1][i] not in colors):
                colors.append(e[ncol-1][i])
                locations.append((ncol-1,i))
    
    locations = list(map(list,locations))
    v = list(itertools.permutations(colors))
    v = list(map(list, v))
    k =[]
    for i in range(len(v)):
        k.append(list(v[i]*16))
        k[i].append(v[i][0])

    for b in range(len(k)):
        count=0
        col = ncol
        t=[[] for _ in range(col)]
        m=0
        for i in range(nrow):
            for j in range(m,col):
                t[i].append(k[b][j])
            m+=nrow
            col +=nrow
    
        for i in range(len(colors)):
            if(t[locations[i][0]][locations[i][1]] == colors[i]):
                count+=1
        if(count==len(colors)):
            return np.array(t)



def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join(".", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

