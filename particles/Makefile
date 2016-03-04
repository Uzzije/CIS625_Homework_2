#
# Hopper - NERSC 
#
# Portland Group Compilers PGI are loaded by default; for other compilers please check the module list
#
CC = g++
MPCC = CC
OPENMP = -mp
CFLAGS = -O3
LIBS =


TARGETS = serial autograder pthreads

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o
pthreads: pthreads.o pthread_barrier.o common.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o pthread_barrier.o common.o
openmp: openmp.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common.o
mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o

autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
pthreads.o: pthreads.cpp common.h
	$(CC) -c $(CFLAGS) pthreads.cpp
mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp
pthread_barrier.o: pthread_barrier.c pthread_barrier.h 
	$(CC) -c pthread_barrier.c

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
