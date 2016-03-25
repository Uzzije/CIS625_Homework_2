# Load CUDA using the following command
# module load cuda

#
# Dirac
#
CC = nvcc
MPCC = nvcc
OPENMP = 
CFLAGS = -O3 -arch sm_20
NVCCFLAGS = -g -arch sm_20
LIBS = -lm

TARGETS = serial gpu

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu || rm -f serial.o
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu || rm -f gpu.o
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu || rm -f common.o

clean:
	rm -f *.o $(TARGETS)
