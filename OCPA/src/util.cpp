#include "util.h"

void writeMatrixData(const Matrix &matrix, std::ostream &out) {
  if (matrix.data == nullptr) {
    out << "Matrix data is null." << std::endl;
    return;
  }

  for (int i = 0; i < matrix.height; ++i) {
    for (int j = 0; j < matrix.width; ++j) {
      out << matrix.data[i * matrix.width + j] << " ";
    }
    out << "\n";
  }
}

void writeMatrixData(const HalfMatrix &matrix, std::ostream &out) {
  if (matrix.data == nullptr) {
    out << "Matrix data is null." << std::endl;
    return;
  }

  for (int i = 0; i < matrix.height; ++i) {
    for (int j = 0; j < matrix.width; ++j) {
      out << __half2float(matrix.data[i * matrix.width + j]) << " ";
    }
    out << "\n";
  }
}

using Width = int;
using Height = int;

std::pair<Width, Height> GetMatrixDimensions(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return {0, 0}; // Return 0,0 if file can't be opened
  }

  std::string line;
  int width = 0, height = 0;

  // Read the first line to determine the width
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    float value;
    while (iss >> value) {
      ++width;
    }
    ++height; // Count the first line
  }

  // Count the remaining lines to determine the height
  while (std::getline(file, line)) {
    if (!line.empty()) {
      ++height;
    }
  }

  file.close();
  return {width, height};
}

bool loadMatrixData(const std::string &filename, Matrix &matrix) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return false;
  }

  std::string line;
  int row = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    float value;
    int col = 0;
    while (iss >> value) {
      if (col < matrix.width && row < matrix.height) {
        matrix.data[row * matrix.width + col] = value;
      }
      ++col;
    }
    ++row;
  }

  OCPA_DEBUG(writeMatrixData(matrix, std::cout));

  file.close();
  return true;
}
bool loadMatrixData(const std::string &filename, HalfMatrix &matrix) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return false;
  }

  std::string line;
  int row = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    float value;
    int col = 0;
    while (iss >> value) {
      if (col < matrix.width && row < matrix.height) {
        matrix.data[row * matrix.width + col] = __float2half(value);
      }
      ++col;
    }
    ++row;
  }

  OCPA_DEBUG(writeMatrixData(matrix, std::cout));

  file.close();
  return true;
}

