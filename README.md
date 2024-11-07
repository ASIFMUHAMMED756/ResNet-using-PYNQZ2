# ResNet 20-using-PYNQZ2

In this project will use the PYNQ Z2 development board and Tensil’s open-source inference accelerator to show how to run machine learning (ML) models on FPGA. We will be using ResNet-20 trained on the CIFAR dataset. These steps should work for any supported ML model – currently, all the common state-of-the-art convolutional neural networks are supported.

`Tensil` is a set of tools for running machine learning models on custom accelerator architectures. It includes an RTL generator, a model compiler, and a set of drivers. It enables you to create a custom accelerator, compile an ML model targeted at it, and then deploy and run that compiled model.`The primary goal of Tensil is to allow anyone to accelerate their ML workloads`. Currently, we are focused on supporting convolutional neural network inference on edge FPGA (field programmable gate array) platforms, but we aim to support all model architectures on a wide variety of fabrics for both training and inference.

![image](https://github.com/user-attachments/assets/d1162f5e-25d7-411e-bd9c-4eada05fab35)

## Step 1
First, we need to get the Tensil toolchain. The easiest way is to pull the Tensil docker container from Docker Hub. The following command will pull the image and then run the container.
```
docker pull tensilai/tensil
docker run -v $(pwd):/work -w /work -it tensilai/tensil bash
```
## Step 2
 The Tensil architecture definition file (.tarch) specifies the parameters of the architecture to be implemented. These parameters are what make Tensil flexible enough to work for small embedded FPGAs as well as large data-center FPGAs. Our example will select parameters that provide the highest utilization of resources on the XC7Z020 FPGA part at the core of the PYNQ Z2 board. The container image conveniently includes the architecture file for the PYNQ Z1 development board at /demo/arch/pynqz1.tarch(note:PynqZ1 and Z2 have same architecture, so it is same for both Pynqz1 and Pynqz2 board)




