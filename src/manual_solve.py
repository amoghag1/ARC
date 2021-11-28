#!/usr/bin/python

# Name: Amogh M. Agnihotri
# Student ID: 21236437
# Course: MSc Artificial Intelligence (1MAI1)
# GitRepo link: https://github.com/amoghag1/ARC

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

"""
Summary/Reflection:
The Multi-dimensional array problems from ARC were solved using python. 
The solutions have been written using a structural programming approach without any use of machine learning module. 
It can be seen in the code the use of nested loops to traverse all elements of the array and perform the necessary operations on them.
A new array is used to store and/or perform the operations and is returned by the function.
The 'copy' module is used to create a deepcopy of the input array to avoid operating on the same array. 
multiple utility functions from the 'Numpy' module are used - np.zeros to create an array of 0's and 
np.full to fill the entire array with a particular value.
np.fill_diagonal is used to fill normal diagonal along with to fill secondary diagonal with flipr arguement.
arrays are traversed with a[i][j] which can be further improved if traversed with a[i, j].
Regular slicing of lists and reshaping of numpy arrays is also used.

"""


"""In task 25d8a9c8 we need to transform x in such a way that rows having all the squares with same colour should be changed to black.
While rest of the rows or squares are to be changed to gray.
First we need to figure out which exact rows have all the squares with same color, save those rows and then change them to black.
Initially I have changed whole input to gray and then printed balck when I found row with all the squares having same color
With my code, all the test and training grids are solved correctly."""
def solve_25d8a9c8(x):
    #creating a deepcopy of x in order to avoid referencing the same variable
    ans = np.zeros(np.shape(x))
    #iterate over length of rows to check if all sqaures in a row are same 
    for i in range(len(x)):
        #with np.all check row by row if sqaures are same
        if(np.all(x[i] == x[i][0])):
            #if all the squares in row are same, fill the row with black color
            ans[i].fill(5)
    return ans

"""In task e179c5f4, we are mostly getting vertical matrices having last square of first column(last row) is painted dark blue.
The conversion that is needed is to draw a zig-zag pattern from the given dark blue column in a diagonal direction in bottom up approach.
First we need to traverse right till the end of columns is reached and then to left till the other end is reached and move 
upwards till we print dark blue sqaure on first column. We know that the each row will have only 1 dark blue square. 
Initially we will start with painting all the the squares light blue and paint over dark blue from left bottom corner.
With my code, all the test and training grids are solved correctly."""
def solve_e179c5f4(x):
    #creating a deepcopy of x in order to avoid referencing the same variable
    a = copy.deepcopy(x)
    #initializing variables to use further
    j=0
    flag = 0
    nrow = a.shape[0]
    nclm = a.shape[1]
    #painting whole a as faint blue
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            a[i][k] = 8
    #traversing from bottom left corner in diagonal way towards left till each row is filled with 1 dark blue square each
    for i in range(nrow):
        #flag 0 to go towards right side of the matrix
        if (flag==0):
            a[nrow-1][j]=1
            nrow-=1
            j+=1
            #as j equals number of columns we flip the flag to start moving towards left from next row
            if(j==nclm):
                j-=2
                flag=1
        #flag 1 to move towards left side of the matrix
        elif (flag==1):
            a[nrow-1][j]=1
            nrow-=1
            j-=1
            #moving left till j becomes -1 and then fliping the flag to move to right on next row
            if(j==-1):
                j=1
                flag=0

    return a

"""In tastk ea786f4a, we are given a n*n matrix with central elemnet colured in black. 
We just need to paint both diagonals of the square in color given in center in this case it is black.
We will just use numpy functions to traverse diagonal, and flip function of same library to traverse through other diagonal.
With my code, all the test and training grids are solved correctly."""
def solve_ea786f4a(x):
    #creating a deepcopy of x in order to avoid referencing the same variable
    b = copy.deepcopy(x)
    #filing main diagonal with black color with np.fill_diagonal
    np.fill_diagonal(b, 0, wrap=True)
    #filing secondary diagonal with black color with np.fill_diagonal and flipr
    np.fill_diagonal(np.fliplr(b), [0])
    return b

"""In task 05269061, in the n*n box, three diagonals from any square of the bottom row with three different colors is given. 
We have to paint 3 diagonals in an order to get the correct output. I considered this more of a horizontal problem rather than diagonal one.
If we observe, we have to fill 49 squares with 3 colors in repeqtation and with correct combination. 
Initially I saved all the unique colors found and their locations except black. I made all possible combinations of those colors in lists of lists.
A single list in list will now have 3 colors which are to be put in 49 boxes.
So I again created 3*16 combinations and added the first color as 49th as it will be the same.
Once I got output with all squares colored I ompared initially saved locations to my output, if all three are correct, we have the answer !
With my code, all the test and training grids are solved correctly.
"""
def solve_05269061(x):
    #creating a deepcopy of x in order to avoid referencing the same variable
    e = copy.deepcopy(x)
    #Creating variables for further use
    (nrow, ncol) = np.shape(e)
    colors = []
    locations = []
    #Getting all the colors and their locations in respective lists
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
    
    #locations changed to list of lists
    locations = list(map(list,locations))
    #Getting permuatations all possible color combinations
    v = list(itertools.permutations(colors))
    v = list(map(list, v))
    #Creating a list of list where each list will contain 49 blocks to be filled with particular order
    k =[]
    for i in range(len(v)):
        k.append(list(v[i]*16))
        #adding last element same as first element
        k[i].append(v[i][0])

    #Painting the input with colors as we have generated above for all combinations
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
        #Comapring created sqaure to all previously saved locations and their respective colors
        for i in range(len(colors)):
            if(t[locations[i][0]][locations[i][1]] == colors[i]):
                count+=1
        #if all three colors match, we return the output as np array        
        if(count==len(colors)):
            return np.array(t)


"""5bd6f4ac: In the task 5bd6f4ac, the nxn mattrix is given which is is composed of random coloured squares.
We have to convert this input in 3*3 matrix which will contain the last three elements of each of first three rows on the sqaure.
With my code, all the test and training grids are solved correctly.
"""
def solve_5bd6f4ac(x):
    #creating a deepcopy of x in order to avoid referencing the same variable
    q = copy.deepcopy(x)
    #initializing list to save the same colors
    saved_colors = []
     #Moving through slices of first three rows and then last three columns
    for i in q[:3]:            
        for j in i[-3:]:
            #saving the color values to list       
            saved_colors.append(j) 
    #converting the list to np array       
    p = np.array(saved_colors) 
    #reshaping p into a 3x3 matrix          
    p = p.reshape(3,3)         

    return p


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


"""
Summary/Reflection:
The Multi-dimensional array problems from ARC were solved using pure python. 
The solutions have been written using a structural programming approach without any use of machine learning module. 
It can be seen in the code the use of nested loops to traverse all elements of the array and perform the necessary operations on them.
A new array is used to store and/or perform the operations and is returned by the function.
The 'copy' module is used to create a deepcopy of the input array to avoid operating on the same array. 
multiple utility functions from the 'Numpy' module are used - np.zeros to create an array of 0's and 
np.full to fill the entire array with a particular value.
np.fill_diagonal is used to fill normal diagonal along with to fill secondary diagonal with flipr arguement.
arrays are traversed with a[i][j] which can be further improved if traversed with a[i, j].
Regular slicing of lists and reshaping of numpy arrays is also used.

"""
