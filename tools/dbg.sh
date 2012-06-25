#!/usr/bin/zsh

for i in dbg/bprop-0*/*.dot ; do dot -Tpng $i > ${i/.dot/.png} ; done 
for i in dbg/fprop-0*/*.dot ; do dot -Tpng $i > ${i/.dot/.png} ; done 
geeqie dbg/fprop-0*/*.png dbg/fprop-0*/*.png
