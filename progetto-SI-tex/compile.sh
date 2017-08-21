biber main
pdflatex -synctex=1 -interaction=nonstopmode %.tex 
biber main 
biber main
ppdflatex -synctex=1 -interaction=nonstopmode %.tex
ppdflatex -synctex=1 -interaction=nonstopmode %.tex 

evince main.pdf 
