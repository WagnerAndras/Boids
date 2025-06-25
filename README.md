# Task
## Boid simulation with extras (up to 125%)
!! The task can only be accepted with real-time graphical display (i.e., OGL/DX/Vulcan interop.).

Implement the boid simulation described on Wikipedia
 - in 2D (20%).
 - in 3D (20%), where the user can navigate space and observe the simulation from different viewpoints (20%).

Enable the user to easily reset and re-parameterize the simulation, setting
 - the number of boids (5%)
 - the FOV of boids (10%)
 - the max perception distance (5%)
 - the weights of boid rules: (5%)
      - separation
      - alignment
      - cohesion
 - the type of initial distribution of boids (e.g., uniform randomization) (10%).

Add obstacles that are avoided by boids (20%).
For greater performance, divide space into blocks, when calculating the direction of boids, start with discarding other boids in blocks outside the perception distance (5%) or FOV (5%).

If you move memory from GPU to CPU before drawing, you lose 40%.


# Documentation
## Hardware specifications used in testing: 
 - Processor: 8 × Intel® Core™ i5-8265U CPU @ 1.60GHz
 - Memory: 7.6 GiB of RAM
 - GPU: NVIDIA GeForce MX250

### CPU
My initial [naive 2D CPU implementation](https://github.com/WagnerAndras/Boids/tree/CPU) runs at approximately 20 FPS with a fixed number of 1024 2D boids in the simulation.

### CUDA
Moving position calculations into [CUDA](https://github.com/WagnerAndras/Boids/tree/cuda) more than doubles this with approximately 50 FPS, even though in every frame the world matrices are copied back to CPU memory before being passed to the vertex shader.

### Interop
Implementing [interoperation](https://github.com/WagnerAndras/Boids/tree/interop) with OpenGL to avoid unnecessary CPU-GPU data exchanges doesn't seem to improve performance in this case.

### Parameters
Boid parameters and initialization settings can be controlled via a [GUI](https://github.com/WagnerAndras/Boids/tree/ui).
In this version I tried moving the boids' data, which is often read, into shared memory, but without any specific scheme to reduce the number of operations by each thread, this lead to no improvements.
Due to the limited amout of shared memory, a much larger number of boids could not be displayed this way, but I deferred further optimizations, as 2048 boids, which I chose as the maximum setting, looks plenty enough in 2D on a 1920 by 1080 display.

### 3D
Switching to a [3D simulation](https://github.com/WagnerAndras/Boids/tree/3D) has very little performance cost, as the algorithm is near exactly the same, and I have been using 4 by 4 world matrices even in 2D to avoid conversions, although changing this could lead to some improvements in the 2D simulation.
