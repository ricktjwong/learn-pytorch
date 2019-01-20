#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:03:56 2018

@author: ricktjwong
"""

import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# We want to solve the equation x dot w = y
# Solve for what w is, given x and y
N, D_in, H, D_out = 3, 3, 5000, 1

# Create random input and output data
w = np.array([[6.],[7.],[8.]])
x = np.array([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
y = np.array([[ 44.],
            [107.],
            [170.]])

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(10000):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # partial L / partial (y_pred - y)
    grad_y_pred = 2.0 * (y_pred - y)
    # partial L / partial w2 = partial L / partial (y_pred - y) * partial (y_pred - y) / partial w2
    grad_w2 = h_relu.T.dot(grad_y_pred)
    # partial L / partial h_relu = partial L / partial (y_pred - y) * partial (y_pred - y) / partial h_relu
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    # partial L / partial w1 = partial L / partial h * partial h / partial w1
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print(y_pred)
# print(x.dot(w))
# print(w)
h = x.dot(w1)
h_relu = np.maximum(h, 0)
y_pred = h_relu.dot(w2)
print(y_pred)
