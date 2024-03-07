# How to install COLMAP under Ubuntu 22.04, CUDA 11.8

<br>
+ [Reference](https://colmap.github.io/install.html)

### Step 1. Install dependencies 

```bash
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    gcc-10 \
    g++-10
```

### Step 2. Install CUDA package

If the system does not come with CUDA, install using 
```bash
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc
```
Then you can find the CUDA library using `ldconfig -p | grep cuda`

Find in the output some path like "/usr/local/cuda-xx/", this means the cuda library is at "/usr/local/cuda-xx/lib64"

```bash
export CUDA_LIB_PATH=/usr/local/cuda-xx/lib64 # find this out!
```

### Step 3. Setup environment vvariables

```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH}"
```

### Step 4. Run CMake config

```bash
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -D CMAKE_CUDA_ARCHITECTURES=75 -GNinja
ninja
sudo ninja install
```

After this, `colmap` should appear under '/usr/local/bin'.