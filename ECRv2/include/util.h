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

bool checkPaths(std::string input) {
  std::ifstream inputStream(input);
  if (!inputStream) {
    std::cerr << "Error: File path '" << input << "' does not exist.\n";
    exit(1);
    return false;
  }
  return true;
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
        OCPA_DEBUG(std::cout << "row: " << row << " col: " << col
                             << " value: " << value << "\n");
        matrix.data[row * matrix.width + col] = value;
      }
      ++col;
    }
    ++row;
  }

  file.close();
  return true;
}

void printMatrixData(const Matrix &matrix) {
  if (matrix.data == nullptr) {
    std::cout << "Matrix data is null." << std::endl;
    return;
  }

  for (int i = 0; i < matrix.height; ++i) {
    for (int j = 0; j < matrix.width; ++j) {
      std::cout << matrix.data[i * matrix.width + j] << " ";
    }
    std::cout << "\n";
  }
}

#endif // UTIL_H
