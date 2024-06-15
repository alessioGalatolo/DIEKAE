export HF_CACHE_DIR=""

for dir in ./data/*     
do
    dir=${dir%*/}      # remove the trailing "/"
    echo
    echo "(${dir##*/})"    # print everything after the final "/"
    python $dir/build.py &
done
wait
