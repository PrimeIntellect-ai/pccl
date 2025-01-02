import os
import sys
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

CMAKE_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Root directory of the CMake project
NUM_JOBS: int = max(multiprocessing.cpu_count() - 1, 1)  # Use all but one core

class BuildException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class CMakeBuildExtension(Extension):
    def __init__(self, name, root_dir: str = ''):
        super().__init__(name, sources=[])
        self.root_dir = os.path.abspath(root_dir)

class CMakeBuildExecutor(build_ext):
    def initialize_options(self):
        super().initialize_options()

    def run(self):
        try:
            print(subprocess.check_output(['cmake', '--version']))
        except OSError:
            raise BuildException('CMake must be installed to build the pccl binaries from source. Please install CMake and try again.')
        super().run()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        output_dir = os.path.abspath(os.path.join(self.build_lib, "pccl"))

        # Get build type
        release_type = "Release"
        if os.environ.get('PCCL_ENABLE_DEBUG_SYMBOLS', None) is not None:
            release_type = "Debug"

        if os.environ.get('PCCL_REL_WITH_DEBUG_SYMBOLS', None) is not None:
            release_type = "RelWithDebInfo"

        # Get C and CXX compiler
        c_compiler = os.environ.get('CC', None)
        cxx_compiler = os.environ.get('CXX', None)

        # Prepare cmake arguments
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={release_type}',  # Specify the build type
        ]
        if c_compiler is not None:
            cmake_args += [f'-DCMAKE_C_COMPILER={c_compiler}']
        if cxx_compiler is not None:
            cmake_args += [f'-DCMAKE_CXX_COMPILER={cxx_compiler}']

        # Probe configure to detect the cmake generator
        print("Probe configure to detect the cmake generator...")
        print(' '.join(['cmake', ext.root_dir] + cmake_args))
        subprocess.check_call(['cmake', ext.root_dir] + cmake_args, cwd=self.build_temp)

        def is_multi_config_generator(generator):
            """Determine if the generator is multi-config."""
            multi_config_generators = [
                'Visual Studio',
                'Xcode',
                'NMake',
                'MSYS Makefiles',
            ]
            return any(gen in generator for gen in multi_config_generators)

        def get_cmake_generator(build_dir):
            cache_file = os.path.join(build_dir, 'CMakeCache.txt')
            if not os.path.isfile(cache_file):
                raise FileNotFoundError(f"CMakeCache.txt not found in {build_dir}")

            with open(cache_file, 'r') as f:
                for line in f:
                    if line.startswith('CMAKE_GENERATOR:INTERNAL='):
                        generator = line.strip().split('=')[-1]
                        return generator
            raise ValueError("CMAKE_GENERATOR not found in CMakeCache.txt")

        # For multi-config generators like Visual Studio
        multi_config_generator_args = [
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={output_dir}',
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG={output_dir}',
        ]

        # For non-multi-config generators like Unix Makefiles
        single_config_generator_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}',
        ]

        # Get the cmake generator
        cmake_generator = get_cmake_generator(self.build_temp)
        print(f"Detected CMake generator: {cmake_generator}")

        if is_multi_config_generator(cmake_generator):
            cmake_args += multi_config_generator_args
        else:
            cmake_args += single_config_generator_args

        # Configure the project
        print("Configuring the project with CMake arguments:")
        print(' '.join(['cmake', ext.root_dir] + cmake_args))
        subprocess.check_call(['cmake', ext.root_dir] + cmake_args, cwd=self.build_temp)

        # Build the project
        build_args = [
            '--target', 'pccl',  # Only build the pccl library
            f'-j{NUM_JOBS}',
            '-v',
            '--config', 'Release',
        ]
        print("Building the project with CMake arguments:")
        print(' '.join(['cmake', '--build', '.'] + build_args))
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Setup dependencies from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

if sys.platform.startswith('win'):
    package_data_files = ['pccl.dll']
elif sys.platform.startswith('darwin'):
    package_data_files = ['libpccl.dylib']
else:
    package_data_files = ['libpccl.so']

# Setup pccl package
setup(
    name='pccl',
    version='0.1.0',
    author='Michael Keiblinger <mike@primeintellect.ai>, Mario Sieg <mario@primeintellect.ai>',
    description='An IP-based collective communications library',
    long_description='An IP-based collective communications library',
    packages=['pccl'],
    package_data={
        'pccl': package_data_files,
    },
    include_package_data=False,
    ext_modules=[CMakeBuildExtension('pccl', root_dir=CMAKE_ROOT)],
    cmdclass={
        'build_ext': CMakeBuildExecutor,
    },
    zip_safe=False,
    install_requires=install_requires
)