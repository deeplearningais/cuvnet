#!/usr/bin/zsh

gp=$(which parallel)
if [ -x "$gp" ]; then
    cd dbg
    find -name '*.dot' | parallel "dot -Tpng {} > {.}.png"
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
