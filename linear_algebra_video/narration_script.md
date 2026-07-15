# Just Enough Linear Algebra for ML — Narration Script

## 01. Just Enough Linear Algebra for ML

Welcome to Just Enough Linear Algebra for Machine Learning. This tutorial is designed for absolute beginners. We will skip the abstract proofs and focus entirely on the load-bearing concepts—vectors, matrices, multiplication rules, transformations, and factorization—that you need to understand how data moves through machine learning models.

## 02. 01. Vectors: The Coordinates of Data

Let us start with vectors. Algebraically, a vector is simply an ordered list of numbers representing features of an observation, such as an house's price and size. Geometrically, it is an arrow in space pointing to specific coordinates. We can add vectors element-wise to combine features, or scale them to adjust magnitudes.

## 03. 02. Matrices: Grids & Datasets

A matrix is a two-dimensional grid of numbers, organized in rows and columns. In machine learning, we stack our data in a design matrix X, where each row represents a single example or observation, and each column represents a single feature or measurement across all examples.

## 04. 03. Matrix Multiplication (GEMM)

Matrix multiplication is the core compute operation in machine learning. To multiply matrix A and B, the number of columns in A must equal the number of rows in B. The resulting cell at row i, column j is the dot product of A's i-th row and B's j-th column. Crucially, matrix multiplication is not commutative: changing the order changes the result.

## 05. 04. Special Matrices: Transpose & Identity

There are several special matrix operations. The transpose, written as X transpose, flips the rows and columns, turning a row-major matrix into a column-major layout. The identity matrix acts like the number one in scalar math: multiplying any matrix by the identity leaves it unchanged. The dot product of two vectors measures their geometric alignment.

## 06. 05. Invertibility & Rank

To solve systems of equations, we use the inverse of a matrix, written as A inverse, which satisfies A times A inverse equals the identity matrix. An inverse exists only for square, full-rank matrices—meaning they have no redundant column directions. If a matrix lacks full rank, it is singular and cannot be inverted.

## 07. 06. Linear Transformations

We can view matrix multiplication geometrically as a linear transformation. A matrix A acts as a function that warps, rotates, or scales space. When we write A times x equals b, we are asking: what input vector x in our domain gets mapped to the target vector b in our codomain? Solving this system is equivalent to inverting the transformation.

## 08. 07. Eigenvectors & Eigenvalues

An eigenvector of a matrix A is a special direction that does not change its orientation when transformed by A; it only gets stretched or shrunk. The scale factor is called the eigenvalue. In machine learning, the leading eigenvectors of a dataset's covariance matrix point in the directions of maximum variance, forming the basis of Principal Component Analysis.

## 09. 08. Singular Value Decomposition (SVD)

Singular Value Decomposition, or SVD, factorizes any rectangular matrix A into three matrices: U, Sigma, and V transpose. U and V transpose contain orthogonal vectors, while Sigma is a diagonal matrix containing singular values that measure scaling magnitudes. By keeping only the largest singular values, we can perform low-rank approximations to compress high-dimensional data.

## 10. Linear Algebra Summary

Let us summarize. Vectors coordinate data features; matrices store datasets. Matrix multiplication computes combinations using row-by-column dot products. Inverting a matrix is solving a geometric projection, which requires full rank. And eigenvectors and SVD identify the principal directions of variance to compress data. With these essentials, you are ready for classical regression and deep learning.
