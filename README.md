# MNIST_demo
SNN simulation for MNIST task

mnist_demo.py: main for SNN simulation
requirements.txt: packages for the environment. Please install bindsNET (which can be downloaded online), pytorch, and numpy.

mnist_file/input_images.npy: input images(28*28),1000 samples, corresponding to firing rate of independent Poission process.
mnist_file/target_labels.npy: label of the images, 1000 samples.
mnist_file/RawMnn.pth: trained parameters in MNN
mnist_file/trained_SNN_parameters.pt: parameters for SNN simulation (converted from RawMnn.pth)

