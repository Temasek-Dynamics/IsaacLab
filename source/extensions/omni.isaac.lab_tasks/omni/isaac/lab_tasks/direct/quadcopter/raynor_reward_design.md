# Reward Design For Traversing the Gate

## Overview

### Observation & Action

$$
\text{observation} = \begin{bmatrix}
    \bold{p}_{gate} \\
    \bold{p}_{goal} \\
    \bold{q}_{drone} \\
    \bold{v}_{drone} \\
    \bold{w}_{drone} \\
    \bold{a}_{k-1} \\
\end{bmatrix} \quad
\text{action} = \mathbf{a}_k = \begin{bmatrix}
    \bold{F}_{drone} \\
    \bold{M}_{drone} \\
\end{bmatrix} \\
\bold{v}_{drone} = \begin{bmatrix}
    v_{x} \\
    v_{y} \\
    v_{z}
\end{bmatrix} \quad
\bold{q}_{drone} = \begin{bmatrix}
    q_{x} \\
    q_{y} \\
    q_{z} \\
    q_{w}
\end{bmatrix} \quad
\bold{w}_{drone} = \begin{bmatrix}
    w_{x} \\
    w_{y} \\
    w_{z}
\end{bmatrix} \\
\bold{p}_{goal} = \begin{bmatrix}
    x_{goal} \\
    y_{goal} \\
    z_{goal}
\end{bmatrix} \quad
\bold{p}_{gate} = \begin{bmatrix}
    \bold{g}_0 \\
    \bold{g}_1 \\
    \bold{g}_2 \\
    \bold{g}_3 \\
\end{bmatrix} \quad
\bold{g}_i = \begin{bmatrix}
    x_{i} \\
    y_{i} \\
    z_{i}
\end{bmatrix} \quad \\
\bold{F}_{drone} = F \quad
\bold{M}_{drone} = \begin{bmatrix}
    M_{x} \\
    M_{y} \\
    M_{z}
\end{bmatrix}
$$

* Observation space: $\mathbb{R}^{29}$, in **body frame**
* Action space: $\mathbb{R}^{4}$
* Action range: $[-1, 1]$

### Reward

The reward for traversing the gate can be mainly divided into three components:

<center>

```mermaid
mindmap
    root((Reward))
        Approaching Gate
        Traversing Gate
        Reaching Goal
```

</center>

* **Approaching Gate**: The agent is rewarded for approaching the gate's center when it is in front of the gate.
* **Traversing Gate**: The agent is rewarded for moving forward through the gate when it is near the gate.
* **Reaching Goal**: The agent is rewarded for reaching the goal position after passing through the gate.

### Penalty

The penalty for traversing the gate can be mainly divided into four components:

<center>

```mermaid
mindmap
    root((Penalty))
        Collision
        Agressive Motion
        Over Time
        Died
```

</center>

* **Collision**: The agent is penalized for **colliding** with the gate.
* **Agressive Motion**: The agent is penalized for moving aggressively, which is defined as a large change in acceleration and too large velocity.
* **Over Time**: The agent is penalized for taking **more time than expected** to traverse the gate.
* **Died**: The agent is penalized for **dying** in the simulation.

### Reward Function

$$
R = R_{approaching} + R_{traversing} + R_{reaching} + P_{collision} + P_{aggressive} + P_{timeout} + P_{died} \\
$$

## Reward Design

### Approaching Gate

The closer the drone is to the gate's center, the bigger the reward will be. The reward is defined as:

$$
R_{approaching} = \alpha * \left[ x_{proj}^k < -l_{traverse} \And \text{not traversed} \right] * (\left| \mathbf{g}_{center}^{k-1} \right| - \left| \mathbf{g}_{center}^k \right|) \\
x_{proj}^k = \mathbf{n}_{gate} \cdot (- \mathbf{g}_{center}^k) \\
\mathbf{n}_{gate} = \frac{\left( \mathbf{g}_{1} - \mathbf{g}_{0} \right) \times \left( \mathbf{g}_{2} - \mathbf{g}_{0} \right)}{\left| \left( \mathbf{g}_{1} - \mathbf{g}_{0} \right) \times \left( \mathbf{g}_{2} - \mathbf{g}_{0} \right)  \right|} \quad
\mathbf{g}_{center} = \frac{\mathbf{g}_{0} + \mathbf{g}_{1} + \mathbf{g}_{2} + \mathbf{g}_{3}}{4}
$$

* $\alpha$: scaling factor for the reward.
* $l_{traverse}$: traversing length in front of the gate.

### Traversing Gate

The more the drone move forward when reaching the gate, the bigger the reward will be. The reward is defined as:

$$
R_{traversing} = \beta * \left[ |x_{proj}^k| \leq l_{traverse} \And |y_{proj}^k| < \frac{w_{traverse}}{2} \And |z_{proj}^k| < \frac{h_{traverse}}{2} \And \text{not traversed}  \right] * (x_{proj}^k - x_{proj}^{k-1} + [\text{traversing}]) \\
y_{proj}^k = \mathbf{n}_{y} \cdot (- \mathbf{g}_{center}^k) \\
z_{proj}^k = \mathbf{n}_{z} \cdot (- \mathbf{g}_{center}^k) \\
\text{Alternatively} \\
R_{traversing} = \beta * \left[ |x_{proj}^k| \leq l_{traverse} \And |y_{proj}^k| < \frac{w_{traverse}}{2} \And |z_{proj}^k| < \frac{h_{traverse}}{2} \right] * (x_{proj}^k - x_{proj}^{k-1} + [\text{first time traversing}] - l_{traverse} * [\text{other traversing}])
$$

* $\beta$: scaling factor for the reward.
* $h_{traverse}$: traversing height of the gate.
* $w_{traverse}$: traversing width of the gate.

!!!note Note
    $w_{traverse}$ should be **less than** $\frac{w_{gate} - w_{drone}}{2}$, and we don't consider the **orientation** of the drone in this reward design.

### Reaching Goal

After passing through the gate, the drone is rewarded for reaching the goal position. The reward is defined as:

$$
R_{reaching} = \gamma * \left[ x_{proj}^k > l_{traverse} \And \text{traversed}  \right] * \left[ \left| \mathbf{p}_{goal}^{k-1} \right| - \left| \mathbf{p}_{goal}^k \right| \right] \\
$$

* $\gamma$: scaling factor for the reward.

### Traversing Check

To reward or not depending on the traversing status, we need to check if the drone has passed through the gate. The traversing status can be checked by the following conditions:

$$
\text{crossed} = (x_{proj}^k > 0 \And x_{proj}^{k-1} < 0) \\
\text{inside} = (|y_{proj}^k| < \frac{w_{gate}}{2} \And |z_{proj}^k| < \frac{h_{gate}}{2}) \\
\text{traversing} = \text{crossed} \And \text{inside} \\
\text{first time traversing} = \left[ \text{traversing} \And \text{not traversed} \right] \\
\text{other traversing} = \left[ \text{traversing} \And \text{traversed} \right] \\
\text{traversed} = \left[ \text{traversing} | \text{traversed} \right] \\
$$

## Penalty Design

### Collision

The drone is penalized for colliding with the gate. The penalty is defined as:

$$
P_{collision} = -\lambda_{collision} * [\text{collision}] \\
\text{collision} := (F_{contact} > 0) \\
$$

* $\lambda_{collision}$: scaling factor for the penalty.
* $F_{contact}$: contact force between the drone and the gate.

### Timeout

The drone is penalized with time going on if it takes too much time. The penalty is defined as:

$$
P_{timeout} = -\lambda_{timeout} * \left[ t > t_{max} \And \text{not traversed} \right] * (t - t_{max}) \\
$$

* $\lambda_{timeout}$: scaling factor for the penalty.
* $t_{max}$: expected maximum time for traversing the gate.

### Aggressive Motion
<!-- TODO -->

The drone is penalized for moving aggressively, which is defined as a large change in acceleration and too large velocity. The penalty is defined as:

$$
P_{aggressive} = P_{jerk} + P_{acceleration} + P_{velocity} \\
P_{jerk} = \lambda_{jerk} \frac{\left| \mathbf{a}_k - \mathbf{a}_{k-1} \right|}{\Delta t} \\
P_{acceleration} = \lambda_{acceleration} \left| \mathbf{a}_k \right| \\
P_{velocity} = \left[ \left| \bold{v}_{drone} \right| > v_{max} \right] * (e^{\lambda_{velocity} ( \left| \bold{v}_{drone} \right| - v_{max})} - 1) \\
$$

### Died

The drone is defined as **died** if it is out of the simulation boundary. The penalty is defined as:

$$
P_{died} = -\lambda_{died} * [\text{died} \And \text{not traversed}] \\
\text{died} := \left[ \bold{p}_{drone} \notin \text{boundary} \right] \\
$$

* $\lambda_{died}$: scaling factor for the penalty.
* $\text{boundary}$: assumed to be a cube including the gate and the goal.
