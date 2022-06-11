#!/usr/bin/sh
dest=pi@172.16.8.100:/home/pi/NN_SSR3
file1=Input_data_max.csv
file2=Output_data_max.csv
file3=disdata_num_weight.csv
file4=history_data_step_num.csv
file5=optimum_weight_1000
file6=optimum_weight_2000
scp $file1 $dest
scp $file2 $dest
scp $file3 $dest
scp $file4 $dest
scp $file5 $dest
scp $file6 $dest
