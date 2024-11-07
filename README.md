# ResNet 20-using-PYNQZ2

### Intelligent Integrated Circuits and Systems Lab, IIIT Kottayam 


In this project we will use the PYNQ Z2 development board and Tensil’s open-source inference accelerator to show how to run machine learning (ML) models on FPGA. We will be using ResNet-20 trained on the CIFAR dataset. These steps should work for any supported ML model – currently, all the convolutional neural networks are supported.

`Tensil` is a set of tools for running machine learning models on custom accelerator architectures. It includes an RTL generator, a model compiler, and a set of drivers. It enables you to create a custom accelerator, compile an ML model targeted at it, and then deploy and run that compiled model.`The primary goal of Tensil is to allow anyone to accelerate their ML workloads`. Currently, we are focused on supporting convolutional neural network inference on edge FPGA (field programmable gate array) platforms, but we aim to support all model architectures on a wide variety of fabrics for both training and inference.

![image](https://github.com/user-attachments/assets/d1162f5e-25d7-411e-bd9c-4eada05fab35)

## Useful Links
* Pynqz2 Board files [https://github.com/xupsh/pynq-supported-board-file]
* Pynq board setup instructions [https://pynq.readthedocs.io/en/v2.3/getting_started.html]
* Tensil AI [https://www.tensil.ai/]

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

Add the verilog files which we got while generating RTL in step 3.


![image](https://github.com/user-attachments/assets/ec1abb18-ccb6-4f97-a82b-d2cfe5882bfe)

Under IP integrator click `Create Block Design`.


Next, click the plus + button in the Block Diagram toolbar (upper left) and select “ZYNQ7 Processing System” (you may need to use the search box). Do the same for “Processor System Reset”. The Zynq block represents the “hard” part of the Xilinx platform, which includes ARM processors, DDR interfaces, and much more. The Processor System Reset is a utility box that provides the design with correctly synchronized reset signals.

Click “Run Block Automation” and “Run Connection Automation”. Check “All Automation”.

Double-click ZYNQ7 Processing System. First, go to Clock Configuration and ensure PL Fabric Clocks have FCLK_CLK0 checked and set to 50MHz.

![image](https://github.com/user-attachments/assets/68d1d554-01dc-4297-b76f-05cacbea0325)

Then, go to PS-PL Configuration. Check `S AXI HP0 FPD`, `S AXI HP1 FPD`, and `S AXI HP2 FPD`. These changes will configure all the necessary interfaces between Processing System (PS) and Programmable Logic (PL) necessary for our design.
![image](https://github.com/user-attachments/assets/1e2db155-1ddb-408a-8b09-d9416245c5d4)

Again, click the plus + button in the Block Diagram toolbar and select “AXI SmartConnect”. We’ll need 4 instances of SmartConnect. First 3 instances (smartconnect_0 to smartconnect_2) are necessary to convert AXI version 4 interfaces of the TCU and the instruction DMA block to AXI version 3 on the PS. The smartconnect_3 is necessary to expose DMA control registers to the Zynq CPU, which will enable software to control the DMA transactions. Double-click each one and set “Number of Slave and Master Interfaces” to 1.

![image](https://github.com/user-attachments/assets/cc1ff18e-ede2-4c82-8b66-425e14e890c3)

Now, connect `m_axi_dram0` and `m_axi_dram1` ports on Tensil block to `S00_AXI` on smartconnect_0 and smartconnect_1 correspondigly. Then connect SmartConnect M00_AXI ports to S_AXI_HP0 and S_AXI_HP2 on Zynq block correspondingly. The TCU has two DRAM banks to enable their parallel operation by utilizing PS ports with dedicated connectivity to the memory.

Next, click the plus + button in the Block Diagram toolbar and select `AXI Direct Memory Access`(DMA). The DMA block is used to organize the feeding of the Tensil program to the TCU without keeping the PS ARM processor busy.

Double-click it. Disable `Scatter Gather Engine` and `Write Channel`. Change `Width of Buffer Length Register` to be 26 bits. Select `Memory Map Data Width` and `Stream Data Width` to be 64 bits. Change `Max Burst Size` to 256.

![image](https://github.com/user-attachments/assets/07e4acf5-ad49-4abc-acfb-8a365dcbf472)

Connect the instruction port on the Tensil top block to the M_AXIS_MM2S on the AXI DMA block. Then, connect M_AXI_MM2S on the AXI DMA block to S00_AXI on smartconnect_2 and, finally, connect smartconnect_2M00_AXI port to S_AXI_HP1 on Zynq.

Connect M00_AXI on smartconnect_3 to S_AXI_LITE on the AXI DMA block. Connect S00_AXI on the AXI SmartConnect to M_AXI_GP0 on the Zynq block.

Finally, click “Run Connection Automation” and check “All Automation”. By doing this, we connect all the clocks and resets. Click the “Regenerate Layout” button in the Block Diagram toolbar to make the diagram look nice.

![image](https://github.com/user-attachments/assets/9050f3e1-e375-4d96-9ad9-32d880a530b7)

Next, switch to the “Address Editor” tab. Click the “Assign All” button in the toolbar. By doing this, we assign address spaces to various AXI interfaces. For example, the instruction DMA (axi_dma_0) and Tensil (m_axi_dram0 and m_axi_dram1) gain access to the entire address space on the PYNQ Z2 board. The PS gains access to the control registers for the instruction DMA.

![image](https://github.com/user-attachments/assets/c8f4136d-e42e-4c50-83d1-3460a69c8b27).

The final step is to create the HDL wrapper for our design, which will tie everything together and enable synthesis and implementation. Right-click the tensil_pynqz1 item in the Sources tab and choose “Create HDL Wrapper”. Keep “Let Vivado manage wrapper and auto-update” selected. Wait for the Sources tree to be fully updated and right-click on tensil_pynqz1_wrapper. Choose Set as Top.

Now it’s time to let Vivado perform synthesis and implementation and write the resulting bitstream. In the Flow Navigator sidebar, click on “Generate Bitstream” and hit OK. Vivado will start synthesizing our Tensil design – this may take around 15 minutes.

## Step 5 

Compile ML model for TCU.The second branch of the Tensil toolchain flow is to compile the ML model to a Tensil binary consisting of TCU instructions, which are executed by the TCU hardware directly. For this tutorial, we will use ResNet20 trained on the CIFAR dataset. The model is included in the Tensil docker image at `/demo/models/resnet20v2_cifar.onnx`. From within the Tensil docker container, run the following command.
```
tensil compile -a /demo/arch/pynqz1.tarch -m /demo/models/resnet20v2_cifar.onnx -o "Identity:0" -s true
```
The resulting compiled files are listed in the ARTIFACTS table. The manifest (tmodel) is a plain text JSON description of the compiled model. The Tensil program (tprog) and weights data (tdata) are both binaries to be used by the TCU during execution. The Tensil compiler also prints a COMPILER SUMMARY table with interesting stats for both the TCU architecture and the model.

## Step 6

Execute using PYNQ.Now it’s time to put everything together on our development board. For this, we first need to set up the PYNQ environment. This process starts with downloading the SD card image for our development board. There’s the detailed instruction for setting board connectivity on the PYNQ documentation website. You should be able to open Jupyter notebooks and run some examples.

Now that PYNQ is up and running, the next step is to scp the Tensil driver for PYNQ. Start by cloning the Tensil GitHub repository to your work station and then copy `drivers/tcu_pynq` to `/home/xilinx/tcu_pynq `onto your board.
```
git clone git@github.com:tensil-ai/tensil.git
scp -r tensil/drivers/tcu_pynq xilinx@192.168.2.99:
```

We also need to scp the bitstream and compiler artifacts.

Next we’ll copy over the bitstream, which contains the FPGA configuration resulting from Vivado synthesis and implementation. PYNQ also needs a hardware handoff file that describes FPGA components accessible to the host, such as DMA. Place both files in /home/xilinx on the development board. Assuming you are in the Vivado project directory, run the following commands to copy files over.

```
scp tensil-pynqz1.runs/impl_1/tensil_pynqz1_wrapper.bit xilinx@192.168.2.99:tensil_pynqz1.bit
scp tensil-pynqz1.gen/sources_1/bd/tensil_pynqz1/hw_handoff/tensil_pynqz1.hwh xilinx@192.168.2.99:
```
Note that we renamed bitstream to match the hardware handoff file name.

Now, copy the .tmodel, .tprog and .tdata artifacts produced by the compiler to /home/xilinx on the board.

```
scp resnet20v2_cifar_onnx_pynqz1.t* xilinx@192.168.2.99:
```
The last thing meeded to run our ResNet model is the CIFAR dataset. You can get it from Kaggle or run the commands below (since we only need the test batch, we remove the training batches to reduce the file size). Put these files in /home/xilinx/cifar-10-batches-py/ on your development board.

```
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xfvz cifar-10-python.tar.gz
rm cifar-10-batches-py/data_batch_*
scp -r cifar-10-batches-py xilinx@192.168.2.99:
```
We are finally ready to fire up the PYNQ Jupyter notebook and run the ResNet model on TCU.

## Step 7

Jupyter notebook
First, we import the Tensil PYNQ driver and other required utilities.

```
import sys
sys.path.append('/home/xilinx')

# Needed to run inference on TCU
import time
import numpy as np
import pynq
from pynq import Overlay
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import pynqz1

# Needed for unpacking and displaying image data
%matplotlib inline
import matplotlib.pyplot as plt
import pickle
```
Now, initialize the PYNQ overlay from the bitstream and instantiate the Tensil driver using the TCU architecture and the overlay’s DMA configuration. Note that we are passing axi_dma_0 object from the overlay – the name matches the DMA block in the Vivado design.
```
overlay = Overlay('/home/xilinx/tensil_pynqz1.bit')
tcu = Driver(pynqz1, overlay.axi_dma_0)
```
Next, let’s load CIFAR images from the test_batch.

```
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

cifar = unpickle('/home/xilinx/cifar-10-batches-py/test_batch')
data = cifar[b'data']
labels = cifar[b'labels']

data = data[10:20]
labels = labels[10:20]

data_norm = data.astype('float32') / 255
data_mean = np.mean(data_norm, axis=0)
data_norm -= data_mean

cifar_meta = unpickle('/home/xilinx/cifar-10-batches-py/batches.meta')
label_names = [b.decode() for b in cifar_meta[b'label_names']]

def show_img(data, n):
    plt.imshow(np.transpose(data[n].reshape((3, 32, 32)), axes=[1, 2, 0]))

def get_img(data, n):
    img = np.transpose(data_norm[n].reshape((3, 32, 32)), axes=[1, 2, 0])
    img = np.pad(img, [(0, 0), (0, 0), (0, tcu.arch.array_size - 3)], 'constant', constant_values=0)
    return img.reshape((-1, tcu.arch.array_size))

def get_label(labels, label_names, n):
    label_idx = labels[n]
    name = label_names[label_idx]
    return (label_idx, name)
```
To test, extract one of the images.
```
n = 7
img = get_img(data, n)
label_idx, label = get_label(labels, label_names, n)
show_img(data, n)
```
You should see the image.

![image](https://github.com/user-attachments/assets/46db11bb-adbb-47a2-8b7b-37f50bfabb6e)

Next, load the tmodel manifest for the model into the driver. The manifest tells the driver where to find the other two binary files (program and weights data).
```
tcu.load_model('/home/xilinx/resnet20v2_cifar_onnx_pynqz1.tmodel')
```
Finally, run the model and print the results! The call to tcu.run(inputs) is where the magic happens. We’ll convert the ResNet classification result vector into CIFAR labels. Note that if you are using the ONNX model, the input and output are named x:0 and Identity:0 respectively. For the TensorFlow model they are named x and Identity.

```
inputs = {'x:0': img}

start = time.time()
outputs = tcu.run(inputs)
end = time.time()
print("Ran inference in {:.4}s".format(end - start))
print()

classes = outputs['Identity:0'][:10]
result_idx = np.argmax(classes)
result = label_names[result_idx]
print("Output activations:")
print(classes)
print()
print("Result: {} (idx = {})".format(result, result_idx))
print("Actual: {} (idx = {})".format(label, label_idx))
```
Here is the expected result:
```
Ran inference in 0.1513s

Output activations:
[-19.49609375 -12.37890625  -8.01953125  -6.01953125  -6.609375
  -4.921875    -7.71875      2.0859375   -9.640625    -7.85546875]

Result: horse (idx = 7)
Actual: horse (idx = 7)
```
## Summary 
In this project we used Tensil to show how to run machine learning (ML) models on FPGA. We went through a number of steps to get here, including installing Tensil, choosing an architecture, generating an RTL design, synthesizing the desing, compiling the ML model and finally executing the model using PYNQ Z2.


















