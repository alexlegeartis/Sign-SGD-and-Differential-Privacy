# Sign-SGD, Heavy-Tailed Noise and Differential Privacy
## Expected results of the development
Modification of Sign-SGD that ensures differential privacy and the proof of its convergence with high probability under heavy-tailed noise.

## Applications
The algorithm could be implemented into programs such as ChatGPT, with the prospect of making more use of the corrupted user data, which in turn, would improve the accuracy of the LLMs.

## Data to test the algorithm
MNIST database: 28x28 black-and-white images of hand-written digits. CIFAR-10 dataset: 60,000 32x32 color images in 10 different classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Quality criteria
The project must meet the following requirements. First, the accuracy and complexity of the algorithm must be on par with existing modifications. Second, a sound proof of correctness of the algorithm and, most importantly, its privacy, must be presented. Increased Byzantine resilience, faster performance, and an absence of parameters to be tuned are also desirable properties of the algorithm.

## Feasibility of the project
The main risk is the possibility of flaws in the proofs, although it could be mitigated by rigorous tests (what is false does not tend to work). Another one is the unacceptable complexity of the algorithm. Indeed, no one can guarantee that the algorithm we seek exists. Provided that the mentioned risks are eliminated, the algorithm must be feasible by its design, especially when applied to the tasks of LLMs.
