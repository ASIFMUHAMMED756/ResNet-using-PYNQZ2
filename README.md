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
 The Tensil architecture definition file (.tarch) specifies the parameters of the architecture to be implemented. These parameters are what make Tensil flexible enough to work for small embedded FPGAs as well as large data-center FPGAs. Our example will select parameters that provide the highest utilization of resources on the XC7Z020 FPGA part at the core of the PYNQ Z2 board. The container image conveniently includes the architecture file for the PYNQ Z1 development board at /demo/arch/pynqz1.tarch(note:PynqZ1 and Z2 have same architecture, so it is same for both Pynqz1 and Pynqz2 board).

 ## Step 3
 Now that we’ve selected our architecture, it’s time to run the Tensil RTL generator. RTL stands for “Register Transfer Level” – it’s a type of code that specifies digital logic stuff like wires, registers and low-level logic. 
 
To generate a design using our chosen architecture, run the following command inside the Tensil toolchain docker container.
 ```
tensil rtl -a /demo/arch/pynqz1.tarch -s true
```

After this step three verilog files will be generated.

## Step 4

In this step we need to open Xilinx Vivado.Before you create new Vivado project you will need to download PYNQ Z1 board definition files from here. Unpack and place them in `/tools/Xilinx/Vivado/2021.2/data/boards/board_files/`.
![image](https://github.com/user-attachments/assets/c5e9fd1f-ee0f-40ed-9386-916025c7aec4.

Under IP integrator click `Create Block Design`.

![image](https://github.com/user-attachments/assets/6907a1f5-9f11-4985-ac44-25254485483e)

Next, click the plus + button in the Block Diagram toolbar (upper left) and select “ZYNQ7 Processing System” (you may need to use the search box). Do the same for “Processor System Reset”. The Zynq block represents the “hard” part of the Xilinx platform, which includes ARM processors, DDR interfaces, and much more. The Processor System Reset is a utility box that provides the design with correctly synchronized reset signals.

Click “Run Block Automation” and “Run Connection Automation”. Check “All Automation”.

Double-click ZYNQ7 Processing System. First, go to Clock Configuration and ensure PL Fabric Clocks have FCLK_CLK0 checked and set to 50MHz.

![image](https://github.com/user-attachments/assets/68d1d554-01dc-4297-b76f-05cacbea0325)

Then, go to PS-PL Configuration. Check `S AXI HP0 FPD`, `S AXI HP1 FPD`, and `S AXI HP2 FPD`. These changes will configure all the necessary interfaces between Processing System (PS) and Programmable Logic (PL) necessary for our design.
![image](https://github.com/user-attachments/assets/1e2db155-1ddb-408a-8b09-d9416245c5d4)

Again, click the plus + button in the Block Diagram toolbar and select “AXI SmartConnect”. We’ll need 4 instances of SmartConnect. First 3 instances (smartconnect_0 to smartconnect_2) are necessary to convert AXI version 4 interfaces of the TCU and the instruction DMA block to AXI version 3 on the PS. The smartconnect_3 is necessary to expose DMA control registers to the Zynq CPU, which will enable software to control the DMA transactions. Double-click each one and set “Number of Slave and Master Interfaces” to 1.

![image](https://github.com/user-attachments/assets/cc1ff18e-ede2-4c82-8b66-425e14e890c3)

Now, connect m_axi_dram0 and m_axi_dram1 ports on Tensil block to S00_AXI on smartconnect_0 and smartconnect_1 correspondigly. Then connect SmartConnect M00_AXI ports to S_AXI_HP0 and S_AXI_HP2 on Zynq block correspondingly. The TCU has two DRAM banks to enable their parallel operation by utilizing PS ports with dedicated connectivity to the memory.

Next, click the plus + button in the Block Diagram toolbar and select “AXI Direct Memory Access” (DMA). The DMA block is used to organize the feeding of the Tensil program to the TCU without keeping the PS ARM processor busy.

Double-click it. Disable “Scatter Gather Engine” and “Write Channel”. Change “Width of Buffer Length Register” to be 26 bits. Select “Memory Map Data Width” and “Stream Data Width” to be 64 bits. Change “Max Burst Size” to 256.













