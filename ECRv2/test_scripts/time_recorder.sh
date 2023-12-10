#!/bin/bash

main() { 
# $1 command line argument represents network type
feature_path=""
kernel_path=""

# file names for resnet features and kernels
resnet_feature="feature_name.txt"
resnet_kernel="kernel_name.txt"


# file names for vgg features and kernels
vgg_feature="all_conv_name.txt"
vgg_kernel="kernel_name.txt"

singleECR_times=""
cuDNN_times=""

# checking whether time files exist if not
# create them under their respective directories
if [ "$1" == "resnet" ]
then

    feature_path="../../dataset/$1/$resnet_feature"
    kernel_path="../../dataset/$1/$resnet_kernel"

    if [ ! -f "../times_resnet/singleECR_times.txt" ]
    then
        singleECR_times="../times_resnet/singleECR_times.txt"
    fi

    if [ ! -f "../times_resnet/cuDNN_times.txt" ]
    then
        cuDNN_times="../times_resnet/cuDNN_times.txt"
    fi

elif [ "$1" == "vggdata" ]
then

    feature_path="../../dataset/$1/$vgg_feature"
    kernel_path="../../dataset/$1/$vgg_kernel"

    if [ ! -f "../times_vgg/singleECR_times.txt" ]
    then 
         singleECR_times="../times_vgg/singleECR_times.txt"
    fi

    if  [ ! -f "../times_vgg/cuDNN_times.txt" ]
    then  
        cuDNN_times="../times_vgg/cuDNN_times.txt"
    fi
fi


# declaring feature_array
declare -a feature_array
# declaring kernel_array
declare -a kernel_array

entry_index=0

# reading feature_name.txt
if [ "$1" == "resnet" ]
then
    while read -r line
    do 
        if [ "$entry_index" != 0 ]
        then
            feature_array[$entry_index]=$line
        fi

        ((++entry_index))

    done < "$feature_path"
elif [ "$1" == "vggdata" ]
then
    while read -r line
    do 
        feature_array[$entry_index]=$line

        ((++entry_index))

    done < "$feature_path"   
fi

# reset the entry_index
entry_index=0

# reading kernel_name.txt
if [ "$1" == "resnet" ]
then
    while read -r line
    do 
        if [ "$entry_index" != 0 ]
        then
            kernel_array[$entry_index]=$line
        fi

        ((++entry_index))

    done < "$kernel_path"
elif [ "$1" == "vggdata" ]
then
    while read -r line
    do 
        kernel_array[$entry_index]=$line

        ((++entry_index))

    done < "$kernel_path"   
fi

time_file="time.txt"
# delete the time file left
# from previous execution
if [ -f $time_file ]
then
    rm "time.txt"
fi

# determining file to write
file_to_write_time=""

# offest for loop for vgg => 0 for resnet => 1
offset=0
if [ "$1" == "vggdata" ]
then 
    offset=0
elif [ "$1" == "resnet" ]
then
    offset=1
fi

for ((i=$offset; i<$entry_index; i++))
do

     cd ../build    
     if [ "$2" == "cudnn" ]
     then   
        ./singleECR --$2 --$3 --kernel ../../dataset/$1/kernel/${kernel_array[$i]} --feature ../../dataset/$1/feature/${feature_array[$i]} --output singleECR_result.txt > $time_file
     elif [ "$2" == "ecr" ]
     then
        ./singleECR --$2 --kernel ../../dataset/$1/kernel/${kernel_array[$i]} --feature ../../dataset/$1/feature/${feature_array[$i]} --output cuDNN_result.txt > $time_file
     fi    
    
    cat $time_file
     # finding out line number in $time_file
     line_no=0
     while read -r line
     do 
        ((++line_no))
     done < "$time_file"

     # determining file to write
     if [ "$2" == "cudnn" ]
     then 

        file_to_write_time=$cuDNN_times
     elif [ "$2" == "ecr" ]
     then
        
        file_to_write_time=$singleECR_times
     fi       

     line_count=$line_no   
     while read -r line
     do 
       if [ "$line_count" == "1" ]
       then
           echo $line >> $file_to_write_time
       fi 
       ((--line_count))
      
     done < "$time_file" 
     cd ../test_scripts
done    
}

# calling main function
main $1 $2 $3