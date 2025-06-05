#load P at end of beam

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

P = 5.422e4
L = 5.3
E = 240e9
I = 6.523e-4

def create_model():
    model = {
        'dense1': tf.keras.layers.Dense(50, activation = 'tanh'),
        'dense2': tf.keras.layers.Dense(50, activation = 'tanh'),
        'dense3': tf.keras.layers.Dense(50, activation = 'tanh'),
        'output_layer': tf.keras.layers.Dense(1)
    }

    return model

def call_model(model, x):
    x = model['dense1'](x)
    x = model['dense2'](x)
    x = model['dense3'](x)
    x = model['output_layer'](x)

    return x

# Governing equation loss

def ge_loss(x, model):
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(x)
        y_pred = call_model(model, x)
        y_x = tape.gradient(y_pred, x)
    y_xx = tape.gradient(y_x, x)
    del(tape)

    # d2y/dx2 = -P(x-L)/EI
    return y_xx + P * (L-x) / (E*I)

def loss(model, x, x_bc, y_bc):
    res = ge_loss(x, model) 

    loss_ge = tf.reduce_mean(tf.square(res))
    bc_y_pred = call_model(model, x_bc)
    loss_bc1 = tf.reduce_mean(tf.square(y_bc - bc_y_pred))

    return loss_ge + loss_bc1

# Defining 1 step of training

def train_step(model, x, x_bc, y_bc, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, x_bc, y_bc)

    grads = tape.gradient(loss_value, [layer.trainable_variables for layer in model.values()])

    grads = [grad for sublist in grads for grad in sublist]
    variables = [var for layer in model.values() for var in layer.trainable_variables]
    optimizer.apply_gradients(zip(grads, variables))

    return loss_value


# Setting up problem ---

# Generating training data

x_train = np.linspace(0, L, 1000).reshape(-1, 1)
x_train = x_train 
x_train = tf.convert_to_tensor(x_train, dtype = tf.float32)

# Boundary conditions for deflection
x_bc = np.array([[0.0], [L]], dtype = np.float32)
y_bc = np.array([[0.0], [-P * (L**3) / (3*E*I)]], dtype = np.float32)
x_bc = tf.convert_to_tensor(x_bc, dtype = tf.float32)
y_bc = tf.convert_to_tensor(y_bc, dtype = tf.float32)

# Defining model

model = create_model()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 1000,
    decay_rate = 0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)


# Training model

epochs = 2500

for epoch in range(epochs):
    loss_value = train_step(model, x_train, x_bc, y_bc, optimizer)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

print('Training complete.')


# Predicting solution

x_test = np.linspace(0, L, 1000).reshape(-1, 1)
x_test = tf.convert_to_tensor(x_test, dtype = tf.float32)
y_pred = call_model(model, x_test).numpy()

y_true = -(P * (3*L*(x_test**2) - x_test**3)) / (6*E*I)


# Plotting result

plt.figure(figsize=(8, 4))
plt.plot(x_test, y_pred, 'b-', label = 'PINN solution')
plt.plot(x_test, y_true, 'r--', label = 'True solution')
plt.xlabel('x')
plt.ylabel('Defelction y')
plt.legend()
plt.title(f'L = {L}m, E = {E}Pa, P = {P}N, I = {I}')
plt.show()

error = np.abs(y_pred - y_true)
plt.figure()
plt.plot(x_test, error, label='Absolute Error')
plt.title('Pointwise Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()
