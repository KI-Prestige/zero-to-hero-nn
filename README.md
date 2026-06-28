# zero-to-hero-nn
Building Neural Networks from scratch following Andrej Karpathy's Zero to Hero series (micrograd → makemore → nanoGPT

Building an architecture on which neural networks can run to power models, tasks and predictions

**Micrograd**:
The engine that powers the network. It enables the network to perform and handle arithmetic operations of all sorts using a Value class. The class contains unique functions, methods and calculations unique to the network. Operations include addition, subtraction, multiplication, exponential, division, and backpropagation. When handling variables, python cannot inherently run arithmetic operations using the basic signs. Given that neural networks require so many of these arithmetic operations, it is required that the engine is built in a way that the network can handle these operations efficiently.

The functions, methods and calculations in the Value class are designed in a way that the network can efficiently utilize various operations in a synchronized manner - the network can call two or more different functions, methods or calculations to perform different operations on a single task or multiple tasks. This is important especially as it relates to backpropagation, which is the most important element of running a neural network.

**Backpropagation**:
Neural networks are designed to receive input data, learn patterns about the data, and give an output. The output varies depending on the task (classification, prediction etc) and as such the network often times must learn patterns about the data over and over again to get the desired output which is why backpropagation is very important. When the network receives input, learns patterns of those inputs and outputs a result, a process known as Forward pass, the network now uses that output (usually a loss function - how badly the network performed) to measure and correct the error in training. It assigns gradients to every step/process/node that led to the final output, starting from the final output and going backwards chronologically, a process known as backward pass. The gradients are calculated using the derivates of 2 pair of nodes (the current and it's preceding node or parent/child). The gradient is what the backpropagation method uses to measure the error of each node and by how much to change the weight of each node in a way that affects the final output. This is where the engine (micrograd) plays a vital role. Each arithmetic operation has a distinct process, formula, method in which the backpropagation is carried out.

**Neural Network**
Once the engine (Micrograd) is built, the network can now efficiently perform it's tasks of receiving inputs, learning patterns, and giving outputs. The process is carried a number of times, as specified before the training begins using backpropagation (forward and backward pass).
The network receives input features. These features are analyzed by neurons - which learn various patterns using predefined rules and conditions to ensure accuracy of results.

**Neurons**
The neurons are embedded in a network. Their task is to receive input data, learn patterns from that data by assigning weights to each data , and send a value as output using an activation method. In practice, each task is carried out by numerous neurons, learning different aspects about the input data. The weights are assigned randomly on the first instance and then changed during backpropagation (backward pass), based on the gradients of each node.

**Layers**
A layer is a collection of neurons. They are compartments that host a specific number of neurons. The purpose of layers is to increase the versatility of the network. In practice once a layer (specific set of neurons) learns patterns from the input features, the result of that learning (training) is passed to the next layer (also a specific set of neurons) to learn more pattern about the data. This is done according the number of layers used in the network until the final result(output).

**Multi-Layer Perception (MLP)**
The MLP  is the model that organizes the layers (collection of neurons) to enable the network learn(train) and perform a task. It defines the input size, number of layers, number of neurons in each layer and the desired output. It then passes these input to the network to enable the network learn patterns from the input, calculate the output (usually a loss function), perform backpropagation, and repeatedly loop through forward and backward pass until we converge to the desired result.

Other factors such as the learning rate (alpha) are set during the calculation of the loss function to help in the accuracy of the training
