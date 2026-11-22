#include "layout/shuffle/options.hpp"
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto options = benchmark::shuffle::Options::Parse(argc, argv);
    return options.help ? 0 : benchmark::shuffle::Run(options);
  } catch (const std::exception &error) {
    std::cerr << "shuffle failed: " << error.what() << '\n';
    return 1;
  }
}
