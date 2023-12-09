#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "ocpa.h"

#ifdef NDEBUG
#define OCPA_DEBUG(x)
#else
#define OCPA_DEBUG(x) x
#endif

template <typename StreamType> bool checkPaths(const std::string &path) {
  StreamType stream(path);
  if (!stream) {
    std::cerr << "Error: Unable to open or create file '" << path << "'.\n";
    return false;
  }
  return true;
}

void writeMatrixData(const Matrix &matrix, std::ostream &out);
void writeMatrixData(const HalfMatrix &matrix, std::ostream &out);

using Width = int;
using Height = int;

std::pair<Width, Height> GetMatrixDimensions(const std::string &filename);

bool loadMatrixData(const std::string &filename, Matrix &matrix);
bool loadMatrixData(const std::string &filename, HalfMatrix &matrix);

#endif // UTIL_H
