# Geo-SWD

Separate the class distributions while preserving their geometry.

An experimental representation-learning approach that learns a transformation
of the feature space by maximizing the distributional separation between two
classes using Sliced Wasserstein Distance (SWD), while simultaneously
preserving the internal pairwise geometry of each class.

---

## Core Idea 
So i've been thinking about,

Can we learn a transformation of the features that makes two classes more distributionally separated while preserving the geometry?

So we Given two class distributions A and B, learn a transformation, T(x) such that it maximizes SWD, max​SWD(T(A),T(B)).
But only SWD maximization using NN would distort the data hence we add some geometry constrains(MSE)
It worked fine on Decision trees but was super bad for others algorithms.

Our Approach:

Given two class distributions:

    A = class 0
    B = class 1

we learn a transformation:

    T(x)

that tries to maximize:

    SWD(T(A), T(B))

while preserving the internal geometry of the classes.

The optimization objective is:

    min_T-SWD(T(A), T(B))+ λ*L_geometry

The negative sign converts maximization into the minimization form used by standard optimizers.

---

## Geometry Preservation

For two samples xi and xj, we want: ||T(xi) - T(xj)|| to be similar to ||xi - xj||

Therefore the geometry loss is:

    L_geometry = MSE of ((d_transformed - d_original))

---

## Entire Pipeline

```text
                Training Data
                     │
                     ▼
             ┌───────────────┐
             │ Standardize   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Class 0 / 1   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ TransformNet  │
             └───────┬───────┘
                     │
            ┌────────┴─────────┐
            │                  │
            ▼                  ▼
       SWD Separation     Geometry Loss
            │                  │
            └────────┬─────────┘
                     ▼
               Optimization
                     │
                     ▼
              Learned Feature
               Representation
                     │
                     ▼
              Decision Tree
                     │
                     ▼
                Prediction
Limitation:
           This is for binary clf only
           This is not the most optimal way
           For Now it works selectively for few models Best for DTs.

Disclaimer: This is not the final version, research is going on.                   
