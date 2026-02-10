from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.gnu import PkgConfigDeps
from conan.errors import ConanInvalidConfiguration
import os

required_conan_version = ">=2.0.0"

class KnowhereConan(ConanFile):
    name = "knowhere"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_diskann": [True, False],
        "with_pageann": [True, False],
        "with_ut": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": False,
        "with_diskann": False,
        "with_pageann": False,
        "with_ut": False,
    }

    def requirements(self):
        # 使用 Conan 2 兼容的包版本
        self.requires("boost/1.85.0")  # 更新到 Conan 2 兼容版本
        self.requires("gflags/2.2.2")
        self.requires("glog/0.6.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("fmt/9.1.0")
        if self.options.with_ut:
            self.requires("catch2/3.3.1")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        if self.settings.compiler.get_safe("cppstd"):
            cxx_std_value = f"c++{self.settings.compiler.cppstd}"
        else:
            cxx_std_value = "c++17"
        tc.variables["CXX_STD"] = cxx_std_value
        tc.variables["WITH_DISKANN"] = self.options.with_diskann
        tc.variables["WITH_PAGEANN"] = self.options.get_safe("with_pageann", False)
        tc.variables["WITH_UT"] = self.options.with_ut
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
