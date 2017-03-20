all:
	g++ -Wall -g main.cpp -I `pkg-config --libs opencv` -o main

