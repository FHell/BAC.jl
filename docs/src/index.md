# Probabilistic behavioural control

## Introduction
A central problem of control theory is to make a system with a complex interior present a unified, aggregate behaviour towards the outside world. Further, the aggregate behaviour should be close to some specification.

We approach this problem by giving a probabilistic notion of what it means to behave close to a specification. This probabilistic notion is based on defining what inputs a system typically encounters and defining a distance of the systems behaviour to the specified behaviour.

In order to make the above notions precise. We will consider systems and specifications given by parametrized input-output differential equations:

```math
\begin{align}
    \mathcal{S} : \dot x(t) &= f(x(t),i(t),p)\\
    o(t) &= g(x(t),p) \nonumber\\
     & \nonumber p \in \mathcal{P}
\end{align}
```

where $x \in X$ is system state, $i \in I$ is the input and $o \in O$ --- the output, typically $\mathbb{R}^n$, and we fix a finite time interval $T$ such that $i(t) \in \mathcal{B}_i \subset I^T$. We assume a fixed initial condition $x(0)$, and that both $f$ and the function space $\mathcal{B}_i$ are chosen such that the solutions of \eqref{eq:system} exist for all of $T$ and are unique functions of the initial conditions. We denote such a system by $\mathcal{S}$. Then the output $o(t)$ depends only on $p$ and $i(t)$. The input and output are the interface between our system and the outside world. Willems behavioural point of view now suggests to not consider the ODE as the primary object of concern but to look instead at the set of possible input-output trajectories, collectively called the behaviour of the system \cite{willems1989models, moor1999supervisory}. The specification then provides such a set of permissible trajectories, and if the input-output trajectories that can occur in the system are contained in this set, the system is said to satisfy the specification.