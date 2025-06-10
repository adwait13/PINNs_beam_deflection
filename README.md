# Physics-Informed Neural Networks for Beam Deflection

This repository contains TensorFlow implementations of Physics-Informed Neural Networks (PINNs) to model deflection in cantilever beams under two different loading conditions:

1. **Point Load at Free End**
2. **Uniform Distributed Load**

The models embed Euler–Bernoulli beam theory into the loss function to ensure physically accurate predictions without relying on labeled data.

---

## 🔧 Project Structure

- `P_point_load` — PINNs implementation for a cantilever beam with a concentrated load at the free end  
- `uniform_distributed_load` — PINNs implementation for a beam with uniform distributed load  

---

## 🧠 Technical Details

- **Governing Equation:** Euler–Bernoulli beam theory  
- **Tools Used:** TensorFlow, NumPy, Matplotlib  
- **PINN Structure:** Fully connected neural network with physics-based loss incorporating differential equation residuals and boundary conditions  

---
