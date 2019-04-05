TENSORFLOW_ROOT = c:/users/mattn/go/src/github.com/tensorflow/tensorflow
TFLITE_OS = windows
CXXFLAGS ?= -I $(TENSORFLOW_ROOT) \
	-I $(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/downloads/flatbuffers/include
LDFLAGS ?= -L $(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/gen/$(TFLITE_OS)_x86_64/lib \
	$(TENSORFLOW_ROOT)/tensorflow/lite/delegates/gpu/libtflite_gpu_gl.a

.PHONY: all clean

all: tflite-sin

tflite-sin: main.cc
	g++ --std=c++11 main.cc -O3 $(CXXFLAGS) $(LDFLAGS) -ltensorflow-lite -lstdc++ -lpthread -ldl -lm -o tflite-sin

clean:
	rm -f lite2
