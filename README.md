# Dependency Installation

## CUDA

http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux

1. Select NVIDIA binary driver from Additional Drivers in Software & Updates
1. Download the toolkit: https://developer.nvidia.com/cuda-downloads. Select correct version based on minimum driver:
    ```
    CUDA 9.1: 387.xx
    CUDA 9.0: 384.xx
    CUDA 8.0  375.xx (GA2)
    CUDA 8.0: 367.4x
    CUDA 7.5: 352.xx
    CUDA 7.0: 346.xx
    CUDA 6.5: 340.xx
    CUDA 6.0: 331.xx
    CUDA 5.5: 319.xx
    CUDA 5.0: 304.xx
    ```
1. Install the repository meta-data, install GPG key, update the apt-get cache, and install CUDA:
    ```
    $ sudo dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb
    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
    $ sudo apt-get update
    $ sudo apt-get install cuda
    ```
1. Reboot the system to load the NVIDIA drivers.
1. Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables (`~/.profile`):
    ```
    PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
    LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
1. Install compatible version of gcc/g++:
    ```
    sudo apt install gcc-6 g++-6
    sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
    sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
    ```
1. Install a writable copy of the samples then build and run the nbody sample:
    ```
    $ cuda-install-samples-9.0.sh ~
    $ cd ~/NVIDIA_CUDA-9.0_Samples/5_Simulations/nbody
    $ make
    $ ./nbody
    ```

## SDL + OpenGL

`$ sudo apt install libsdl2-dev`
`$ sudo apt install libsdl2-ttf-dev`
`$ sudo apt install libglew-dev`
