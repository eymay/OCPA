"""
    argparse module is used to
    parse command line arguments

    scipy.signal mdoule's convolve
    method is used to test convolution
    results
 """

import argparse
import numpy as np
from scipy.signal import convolve

# function to construct matrices (either feature or kernel)
def construct_matrix(file_desc,empty_mat):
      for line in file_desc:
        line = line.rstrip()
        line =line.split()
        empty_mat.append(line)

def convert_entries_to_float(mat):
     # converting entries to floating-point values
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            mat[row][col] = float(mat[row][col])

# if the difference between corresponding entries 
# is less than some epsilon (0.15) than difference
# can be acceptable
def check_for_numeric_difference(conv_result_mat,numpy_result_mat):
    max_error = 0
    for row in range(len(conv_result_mat)):
            for col in range(len(conv_result_mat[row])):
                if (abs(conv_result_mat[row][col])-abs(numpy_result_mat[row][col])) >= 0.15:
                    return False    
                if abs(conv_result_mat[row][col])-abs(numpy_result_mat[row][col]) > max_error:
                    max_error = abs(conv_result_mat[row][col])-abs(numpy_result_mat[row][col])      
                           

    return True,max_error


def main():
    parser = argparse.ArgumentParser()
    # adding an CLI option for kernel path
    parser.add_argument("-k","--kernel", help="give a path to kernel")
    # adding an CLI option for feature path 
    parser.add_argument("-f","--feature",help="give a path to features")
    # adding CLI option for convolution result obtained from CUDA
    parser.add_argument("-t","--test_output",help="give a path to convolution result")
    args = parser.parse_args()

    
    if args.kernel is None: 
        print("Usage is: python3 --kernel kernel_path --feature feature_path --test_output test_path")
        print("kernel path must be provided")
    
    elif args.feature is None:
        print("feature path must be provided")
        print("Usage is: python3 --kernel kernel_path --feature feature_path --test_output test_path")
   
    elif args.test_output is None:
        print("convolution result path must be provided")
        print("Usage is: python3 --kernel kernel_path --feature feature_path --test_output test_path")
    
    else:
        kernel_path = args.kernel
        feature_path = args.feature
        result_path = args.test_output
        
        # reading kernel into 2-D array (matrix)
        kernel_matrix = []
        kernel = open(kernel_path,"r")
        construct_matrix(kernel,kernel_matrix)

        kernel.close()

        # converting entries to floating-point values
        convert_entries_to_float(kernel_matrix)
        numpy_kernel_matrix = np.array(kernel_matrix)
        numpy_kernel_matrix.resize(3,3)

        # reading features into 2-D array (matrix)
        feature_matrix = []
        features = open(feature_path,"r")
        # filling feature matrix
        construct_matrix(features,feature_matrix)

        features.close()
        row_no = len(feature_matrix)
        col_no = len(feature_matrix[0])
        # converting entries to floating-point values
        convert_entries_to_float(feature_matrix)
        numpy_feature_matrix = np.array(feature_matrix)
        numpy_feature_matrix.resize(row_no,col_no)
    
        # finding the convolution numpy_feature matrix and numpy_kernel_matrix
        conv_result = convolve(numpy_feature_matrix,numpy_kernel_matrix)

        result_matrix = []
        result = open(result_path,"r")
        construct_matrix(result,result_matrix)
        # converting entries to floating-point values
        convert_entries_to_float(result_matrix)
        numpy_result_matrix = np.array(result_matrix)

        # resizing the conv_matrix with respect 
        # numpy result matrix
        conv_result.resize(len(numpy_result_matrix),len(numpy_result_matrix[0]))
        # calling the test function 
        test_result,max_error = check_for_numeric_difference(conv_result,numpy_result_matrix)   

        print("\nMAX ERROR IS: ",max_error,"\n")

        if (test_result):
            print("CONVOLUTION TEST PASSED!!")
        else:
            print("CONVOLUTION TEST FAILED!!")    

main()