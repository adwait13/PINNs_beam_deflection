#linear load from q1 -> q2

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# To ensure the network gives consistent op everytime, otherwise every time it runs it gives different outputs

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

q1 = 40e3
q2 = 20e3
L = 4
E = 1
I = 1.006e6

def create_model():
    model = {
        'dense1': tf.keras.layers.Dense(64, activation = 'tanh'),
        'dense2': tf.keras.layers.Dense(64, activation = 'tanh'),
        'dense3': tf.keras.layers.Dense(64, activation = 'tanh'),
        'dense4': tf.keras.layers.Dense(64, activation = 'tanh'),
        'output_layer': tf.keras.layers.Dense(1, dtype = 'float64')
    }

    return model

def call_model(model, x):
    x = model['dense1'](x)
    x = model['dense2'](x)
    x = model['dense3'](x)
    x = model['dense4'](x)
    x = model['output_layer'](x)

    return x

# Governing equation loss

def ge_loss(x, model):
    with tf.GradientTape(persistent= True) as tape1:
        tape1.watch(x)
        with tf.GradientTape(persistent= True) as tape2:
            tape2.watch(x)
            with tf.GradientTape(persistent= True) as tape3:
                tape3.watch(x)
                with tf.GradientTape(persistent= True) as tape4:
                    tape4.watch(x)
                    y = call_model(model, x)
                y_x = tape4.gradient(y, x)
            y_xx = tape3.gradient(y_x, x)
        y_xxx = tape2.gradient(y_xx, x)
    y_xxxx = tape1.gradient(y_xxx, x)
    
    del tape1, tape2, tape3, tape4 

    rhs = q1 + (q2 - q1) * x /L

    # EI (d4 y/ dx4) = q1 + (q2-q1/L) * x
    return y_xxxx + rhs / (E*I)

def loss(model, x, x_bc, y_bc):
    res = ge_loss(x, model) 

    #loss from governing equation

    loss_ge = tf.reduce_mean(tf.square(res))
    bc_y_pred = call_model(model, x_bc)

    x0 = tf.convert_to_tensor([[0.0]], dtype=tf.float64) # x = 0
    
    #boundary condition at x=0: y(0) = y'(0) = 0

    with tf.GradientTape() as tape:
        tape.watch(x0)
        y_pred = call_model(model, x0)

    y_x0 = tape.gradient(y_pred, x0)

    loss_bc1 = tf.reduce_mean(tf.square(y_bc - bc_y_pred))
    loss_bc2 = tf.reduce_mean(tf.square(y_x0))

    xL = tf.convert_to_tensor([[L]], dtype=tf.float64)  # x = L

    # boundary conditions at x=L: y''(L) = y'''(L) = 0
    
    with tf.GradientTape(persistent= True) as tape1:
        tape1.watch(xL)
        with tf.GradientTape(persistent= True) as tape2:
            tape2.watch(xL)
            with tf.GradientTape(persistent= True) as tape3:
                tape3.watch(xL)
                y_pred = call_model(model, xL)
            y_x = tape3.gradient(y_pred, xL)
        y_xx = tape2.gradient(y_x, xL)
    y_xxx = tape1.gradient(y_xx, xL)

    del tape1, tape2, tape3

    loss_bc3 = tf.reduce_mean(tf.square(y_xx))
    loss_bc4 = tf.reduce_mean(tf.square(y_xxx))
    
    return loss_ge + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4


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

x_train = np.linspace(0, L, 500).reshape(-1, 1)
x_train = x_train 
x_train = tf.convert_to_tensor(x_train, dtype = tf.float64)

# Boundary conditions for deflection
x_bc = np.array([[0.0]], dtype = np.float64)
y_bc = np.array([[0.0]], dtype = np.float64)
x_bc = tf.convert_to_tensor(x_bc, dtype = tf.float64)
y_bc = tf.convert_to_tensor(y_bc, dtype = tf.float64)

# Defining model

model = create_model()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 3000,
    decay_rate = 0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)


# Training model

epochs = 20000

for epoch in range(epochs):
    loss_value = train_step(model, x_train, x_bc, y_bc, optimizer)
    
    if loss_value < 1e-7:
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

print(f"Epoch {epochs}: Loss = {loss_value.numpy()}")

print('Training complete.')


# Predicting solution

x = np.linspace(0, L, 1000).reshape(-1, 1)
x = tf.convert_to_tensor(x, dtype = tf.float64)
y_pred = call_model(model, x).numpy()

y1 = q2 * x**2 / (24 * E * I) * (6 * L**2 - 4 * L * x + x**2)
y2 = ((q1-q2) * x**2) / (120 * L * E * I) * (10 * L**3 - 10 * L**2 * x + 5 * L * x**2 - x**3)
y_true = -(y1 + y2)

# Plotting result

plt.figure(figsize=(8, 4))
plt.plot(x, y_pred, 'b-', label = 'PINN solution')
plt.plot(x, y_true, 'r--', label = 'True solution')
plt.xlabel('x')
plt.ylabel('Defelction y (m)')
plt.legend()
plt.title(f'L = {L}m, E = {E}Pa, q1 = {q1}N, q2 = {q2}N, I = {I}')
plt.show()
plt.savefig("deflection_plot.png", dpi=300, bbox_inches='tight')
