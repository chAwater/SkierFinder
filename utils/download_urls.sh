#!/bin/bash
###############################################################################
# 
# Author: hoho2b
# Created Time: 2019年04月04日 星期四 15时20分30秒
# 
###############################################################################

file='pw_20190331_wl_n1.csv'
while read line
do
    # Get name
    nid=`echo $line | cut -d, -f 1`
    url=`echo $line | cut -d, -f 2`
    # Download
    wget -c -nv $url -O $nid.jpg
    # Random sleep
    s=`date +%s%N | cut -c12-13`
    sleep $s
done < $file

