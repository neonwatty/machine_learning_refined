#!/bin/bash

if [[ $1 == "-all" ]]; then # re-make all html files
	find presentations -type f -print0 | while read -d $'\0' file; do		
  		# lecture files: ipynb and Lecture and NOT checkpoint 	
  		if [[ $file == *"ipynb"* ]] && [[ $file != *"checkpoint"* ]]; then
  			# convert to html
  			eval "jupyter nbconvert --to slides --reveal-prefix=../revealjs" $file
		fi
	done
fi


