#!/bin/bash

if [[ $1 == "-all" ]]; then # re-make all html files

	find notes -type f -print0 | while read -d $'\0' file; do
		# blog_post files: ipynb and NOT Lecture and NOT checkpoint 
		if [[ $file == *"ipynb"* ]] && [[ $file != *"Lecture"* ]] && [[ $file != *"checkpoint"* ]]; then
  			# convert to raw html
  			eval "jupyter nbconvert --to html" $file
  			# make it pretty!
  			eval "python html/prettify.py" ${file/.ipynb/.html}
  		fi  		
  		# lecture files: ipynb and Lecture and NOT checkpoint 	
  		if [[ $file == *"ipynb"* ]] && [[ $file == *"Lecture"* ]] && [[ $file != *"checkpoint"* ]]; then
  			# convert to html
  			eval "jupyter nbconvert --to slides --reveal-prefix=revealjs" $file
		fi
	done

else

	# get modified files
	unstaged_files=$(git diff --name-only 2>&1)
	# get new files
	untracked_files=$(git ls-files --other --exclude-standard 2>&1)

	for file in $unstaged_files $untracked_files; do 
		# blog_post files: ipynb and NOT Lecture and NOT checkpoint 
		if [[ $file == *"ipynb"* ]] && [[ $file != *"Lecture"* ]] && [[ $file != *"checkpoint"* ]]; then
  			# convert to raw html
  			eval "jupyter nbconvert --to html" $file
  			# make it pretty!
  			eval "python html/prettify.py" ${file/.ipynb/.html}
  		fi  		
  		# lecture files: ipynb and Lecture and NOT checkpoint 	
  		if [[ $file == *"ipynb"* ]] && [[ $file == *"Lecture"* ]] && [[ $file != *"checkpoint"* ]]; then
  			# convert to html
  			eval "jupyter nbconvert --to slides --reveal-prefix=revealjs" $file
		fi
	done 
fi