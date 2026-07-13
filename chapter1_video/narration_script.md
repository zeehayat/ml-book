# Chapter 1 Video Tutorial — Narration Script

## 01. The Anatomy of a Tensor

Welcome to Chapter One: The Anatomy of a Tensor and Compute Hardware. In this tutorial, we will look beneath NumPy and PyTorch and build a hardware-level mental model of a tensor. By the end, shape, strides, slicing, transpose, contiguity, and memory performance should feel like consequences of one simple idea: multidimensional data lives in a flat buffer. Keep one question in mind throughout the lesson: when I ask for a logical element, which physical memory address must the computer read?

## 02. What you will be able to do

Our goal is not to memorize framework behavior. It is to predict it. First, we will identify the small pieces of metadata that turn a byte buffer into a tensor. Then we will derive the flat-index formula and use it to understand row-major and column-major layouts. Next, we will transform metadata to create slices, transposes, and reversed views without moving data. Finally, we will connect memory layout to CPU caches, GPU bandwidth, implementation choices, and the most common production mistakes.

## 03. Why not nested Python lists?

Suppose we store ten thousand grayscale images as nested Python lists. A lookup such as images three forty-seven, row five, column thirteen does not jump directly to a pixel. Python follows a pointer to an image list, another pointer to a row list, and another pointer to a Python number object. Each number also carries object metadata. A tensor removes that overhead. It stores the numerical values, and only the values, in one flat block. A few integers then describe the logical geometry. The result is smaller, predictable, and friendly to hardware that reads consecutive bytes efficiently.

## 04. A tensor is a small contract

At hardware level, a tensor is a contract between a flat buffer and some metadata. The buffer owns the raw values. The dtype says how to interpret bytes and how wide each element is. Shape gives the length of every logical axis. Strides say how far to jump in the buffer when an index increases by one on each axis. Finally, a base offset says where this particular view begins. Frameworks may add devices, gradients, and dispatch machinery, but this is the load-bearing representation.

## 05. One formula explains every layout

Here is the chapter's central equation. To translate a multidimensional index, start at the view's offset. For every axis, multiply that axis index by its stride, then add the results. For a three by four row-major matrix, the strides are four and one. Index two, one maps to zero plus two times four plus one times one, which is buffer position nine. The boundary rule is equally important: every index on axis k must be at least zero and less than shape k. Shape controls valid logical indices; strides control physical movement.

## 06. Row-major: the last axis moves fastest

In C-contiguous, or row-major, order, the last axis moves fastest. Our three by four matrix has stride one for columns because moving one column means moving one buffer element. The row stride is four because moving down one row skips the four elements in the previous row. In general, the final stride is one, and each earlier stride is the product of all dimensions to its right. Notice the highlighted path: walking left to right through a row visits consecutive addresses. That access pattern is ideal for cache lines.

## 07. Column-major: the first axis moves fastest

Fortran-contiguous, or column-major, order uses the same buffer model with different strides. Now the row stride is one and the column stride is three. Walking downward through a column visits consecutive addresses. In general, the first stride is one, and each later stride is the product of dimensions to its left. Row-major and column-major are therefore not different kinds of tensor. They are two particular stride configurations. This is why one strided representation can support C, Fortran, NumPy, PyTorch, and many layouts in between.

## 08. Slicing is an affine metadata update

A basic slice does not need to copy values. On a sliced axis, the new view starts later, so its offset becomes the old offset plus start times the old stride. If the step is greater than one, the new stride becomes the old stride times that step. The new shape is simply the number of selected positions. In this example, selecting every second column changes the column stride from one to two. Both the original tensor and the view still point to the same buffer, so writing through the view changes the original data.

## 09. Transpose swaps metadata, not bytes

Transpose is even simpler. For a two-dimensional tensor, swap the two shape entries and swap the two stride entries. A three by four tensor with strides four and one becomes a four by three view with strides one and four. The buffer remains untouched. That is why transposing a gigabyte-scale tensor can complete almost instantly: the operation changes a handful of integers, not a billion bytes. The tradeoff appears later, when a downstream operation traverses those bytes in an unfavorable order or requires contiguous input.

## 10. Reshape asks a stricter question

Reshape is not merely changing the labels on axes. A metadata-only reshape is valid only if the new shape can describe the same logical address sequence without rearranging storage. A contiguous tensor usually allows many reshapes because its addresses form one dense progression. A transposed view often does not. In PyTorch, view enforces this contiguity contract and may reject the operation. Reshape is more permissive: it returns a view when possible and silently allocates a contiguous copy when necessary. This difference matters in performance-sensitive loops.

## 11. Negative strides reverse a view

A negative stride means that increasing a logical index moves backward in physical storage. To reverse a five-element vector without copying, begin at buffer position four, use shape five, and use stride negative one. Logical index zero reads position four; logical index four reads position zero. The subtle point is validation. Checking only the starting offset is unsafe. An implementation must compute the minimum and maximum reachable offsets across the entire shape, accounting for every positive and negative stride, and confirm that both lie inside the backing buffer.

## 12. Why access order changes speed

Modern processors hide slow memory behind a hierarchy. Registers are tiny and fastest. L1 and L2 caches are small and close to each core. L3 is larger and slower. DRAM is vastly larger but can cost well over one hundred cycles to reach. When the CPU requests one float, hardware typically fetches a whole sixty-four-byte cache line, enough for sixteen float thirty-two values. If the next fifteen values are adjacent, that one miss pays for many useful reads. If a large stride jumps to a different cache line each time, most fetched bytes go unused.

## 13. Contiguity is a performance property

Compare these two walks. The contiguous walk consumes neighboring values from every fetched cache line. The strided walk touches one value, skips the rest of the line, and forces another memory request. Both loops perform the same number of additions, so big-O notation calls both linear. Hardware does not. Average memory access time can be modeled as hit time plus miss rate times miss penalty. A worse miss rate can create a large constant-factor slowdown. This is why loop order should match layout: iterate along the stride-one axis in the innermost loop.

## 14. When should you call contiguous()?

Calling contiguous materializes values in a new dense buffer. That costs linear time, consumes extra memory, and uses memory bandwidth. The copy is worthwhile only when its one-time cost is smaller than the repeated penalty of operating on the strided layout. For a single cheap traversal, keep the view. For many kernels, or for an operation whose implementation is dramatically faster on dense input, copying once may win. Treat contiguous as an explicit performance tradeoff, not a ritual. Profile the complete workload, including allocation and transfer costs.

## 15. The same story continues on the GPU

A GPU has its own hierarchy. Large global HBM holds tensors. An on-device L2 cache reduces some traffic. Each streaming multiprocessor offers much faster shared memory for cooperating threads, plus registers for individual threads. Efficient kernels load global memory in coalesced, neighboring transactions, reuse tiles from shared memory, and keep intermediate values in registers. Strides influence whether neighboring threads request neighboring addresses. A logically correct layout can still waste most of a memory transaction if its access pattern is scattered.

## 16. Arithmetic intensity predicts the bottleneck

Arithmetic intensity is the number of useful operations performed per byte moved from memory. Elementwise addition has low intensity: it reads inputs, performs one addition, and writes an output. It is usually bandwidth-bound. Matrix multiplication can reuse loaded tiles for many multiply-adds, giving much higher intensity and a better chance of becoming compute-bound. Layout optimization matters because coalescing, caching, fusion, and reuse reduce unnecessary byte movement. Peak floating-point throughput is irrelevant when the kernel cannot feed the arithmetic units quickly enough.

## 17. A minimal strided tensor implementation

A minimal implementation needs a buffer, shape, strides, and offset. Element access validates the number and bounds of indices, computes the flat position with the central formula, and reads or writes the buffer. Slice and transpose return a new object that shares the same buffer but carries transformed metadata. Iteration must generate logical indices and translate each one. A robust implementation also handles zero-length dimensions, negative strides, axis insertion, and reachable-range validation. NumPy and PyTorch add optimized kernels and device machinery, but their view semantics grow from this same skeleton.

## 18. Five mistakes to catch early

Here are the mistakes that cause the most confusion. First, a basic slice usually aliases the original buffer, so mutation is shared. Second, reshape may allocate when the address pattern is incompatible. Third, C-order strides multiply dimensions to the right, while Fortran order multiplies dimensions to the left. Fourth, negative strides require validating the whole reachable range. Fifth, byte offsets depend on dtype width, and performance depends on traversal order. When debugging, print shape, strides, offset, dtype, and whether the tensor is contiguous before inspecting values.

## 19. The mental model to keep

Let us compress the chapter into five statements. A tensor is a logical view over a flat typed buffer. The physical address is the offset plus the dot product of indices and strides. Basic slicing and transpose usually create views by transforming metadata. Reshape is zero-copy only when the existing address sequence supports the new geometry. And layout is not cosmetic: it controls cache-line use, GPU coalescing, and memory bandwidth. With this foundation, automatic differentiation in Chapter Two becomes easier to reason about because every value and gradient ultimately lives in one of these strided buffers.
