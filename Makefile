CC=nvcc

DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g -G -DDEBUG=1
else
CFLAGS=-O3 -g -lineinfo
endif

CFLAGS+=-Xcompiler=-Wall -maxrregcount=32 -arch=sm_75
CFLAGS+=`pkg-config libibverbs --cflags --libs`
# Use to find out shared memory size
# CFLAGS+=--ptxas-options=-v 

FILES=server client

all: $(FILES)

server: ex3.o server.o common.o
	nvcc --link $(CFLAGS) $^ -o $@
client: ex3.o client.o common.o ex3-cpu.o
	nvcc --link $(CFLAGS) $^ -o $@

server.o: server.cu ex3.h ex2.h
client.o: client.cu ex3.h
common.o: common.cu ex3.h
ex3.o: ex3.cu ex3.h ex2.cu ex2.h
ex3-cpu.o: ex3-cpu.cu ex3.h

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
