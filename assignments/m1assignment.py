# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:50:05 2024

CMPSC445 - M1 Assignment 

@author: dvs6026
"""

def matrixMultiply(A, B):
    
    # dimensions
    Arows = len(A)
    Acols = len(A[0])
    Bcols = len(B[0])
       
    # product init
    AB = [[0] * Bcols for i in range(Arows)]
    
    # matrix multiplication
    for i in range(Arows):
        for j in range(Bcols):
            AB[i][j] = sum(A[i][k] * B[k][j] for k in range(Acols))

    return AB    



if __name__ == "__main__":
    
    # list comprehension
    M = [[3 for i in range(90)] for i in range(100)]
    O = [[i for i in range(100)] for i in range(90)]
    
    n = 5
    
    Mn = [i[:] for i in M]
    Mn[0] = [i*n for i in Mn[0]]
    
    Mo = matrixMultiply(M, O)