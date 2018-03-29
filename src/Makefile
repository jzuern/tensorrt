CC=g++
CFLAGS=-O2 -std=c++0x -I. -I/usr/local/cuda-8.0/include -I/home/jannik/Desktop/hiwi/TensorRT-3.0.4/include


LIBDIRS= -L/usr/local/cuda-8.0/lib64 -L/home/jannik/Desktop/hiwi/TensorRT-3.0.4/lib

LDFLAGS=$(LIBDIRS) -lm -lstdc++ -lnvinfer -lnvparsers -lcuda -lcublas -lcurand -lcudart

DEPS = tensorrt.hpp

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: tensorrt

tensorrt: tensorrt.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)


.PHONY: clean

clean:
	rm -f *.o tensorrt