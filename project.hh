#ifndef PROJECT_HH
#define PROJECT_HH

#include <chrono>
#include <filesystem>

#include "lab.hh"

namespace utcn::ip {
class Project : public Lab {
  static inline std::map<int, std::string> LAB_MENU = {{1, "Run project"}};

  static void testProject();

 public:
  void runLab() override;
};
}  // namespace utcn::ip

#endif  // LAB1_HH