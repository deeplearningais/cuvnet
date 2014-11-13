#!/usr/bin/zsh

if [[ "$1" == "--help"  ||  "$1" == ""  ||  "$1" == "-h" ]]; then
    echo "Usage: $0 {path}"
    echo "  where path is the base path to the experiment, containing the dbg folder"
    exit 0
fi

gp=$(which parallel)
if [ -x "$gp" ]; then
    cd $1/dbg
    find -name '*.dot' | parallel --gnu "dot -Tpng {} > {.}.png"
    #for i in dbg/bprop-0*/*.dot ; do dot -Tpng $i > ${i/.dot/.png} ; done 
    #for i in dbg/fprop-0*/*.dot ; do dot -Tpng $i > ${i/.dot/.png} ; done 
    if [ -d bprop-000 ] ; then
        geeqie fprop-*/*.png bprop-*/*.png
    else
        geeqie fprop-*/*.png 
    fi
else
    echo "Please install GNU parallel, e.g. sudo aptitude install parallel"
fi
