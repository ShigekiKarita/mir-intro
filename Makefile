mir-intro: source/app.d
	dub build --build=release-nobounds --compiler=ldmd2 --build-mode=singleFile --parallel

result.csv: mir-intro
	./mir-intro > result.csv

plot.png: result.csv plot.py
	python plot.py

all: plot.png
