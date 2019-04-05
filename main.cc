#include <vector>
#include <chrono>
#include <iostream>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
using namespace std;

bool
is_error(TfLiteStatus const & status) {
  return status != kTfLiteOk;
}

int
main(int argc, char const * argv[]) {
  std::string a = "sin_model.tflite";
  TfLiteStatus status;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::cout << "Loading model: " << a << std::endl;
  model = tflite::FlatBufferModel::BuildFromFile(a.c_str());
  if (!model) {
	std::cerr << "Failed to load the model." << std::endl;
	return -1;
  }
  std::cout << "The model was loaded successful." << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(* model, resolver)(& interpreter);
  std::cout << "interpreter was build successful." << std::endl;
  auto* delegate = TfLiteGpuDelegateCreate(nullptr);
  if (interpreter->ModifyGraphWithDelegate(delegate) == kTfLiteOk) {
	std::cerr << "Failed to enable GPU." << std::endl;
	return -1;
  }
  status = interpreter->AllocateTensors();
  if (is_error(status)) {
	std::cerr << "Failed to allocate the memory for tensors." << std::endl;
	return -1;
  }
  std::cout << "The model was allocated successful." << std::endl;
  ofstream fout("pred.csv");
  float * in = interpreter->typed_input_tensor<float>(0);
  float * out = interpreter->typed_output_tensor<float>(0);
  fout << "x,y,p" << endl; 
  int i;
  double x, y;
  double pi = acos(-1.0);
  for (i = -20; i < 20; i++) {
	x = (double) i / 6;
	y = sin(x);
	in[0] = x;
	status = interpreter->Invoke();
	if (is_error(status)) {
	  std::cerr << "Failed to invoke the interpreter." << std::endl;
	  return -1;
	}
	std::printf ("%2.2f\n", out[0]);
	fout << x << "," << y << "," << out[0] << endl;
  }
  cout << "ok" << endl;
  fout.close();
  return 0;
}
