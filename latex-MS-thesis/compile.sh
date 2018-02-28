biber main
pdflatex -synctex=1 -interaction=nonstopmode %.tex 
biber main 
biber main
pdflatex -synctex=1 -interaction=nonstopmode main.tex
pdflatex -synctex=1 -interaction=nonstopmode main.tex 

evince main.pdf 
