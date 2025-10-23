# Scientific Computation in Biophysics

## Overview

This repository contains a computational biophysics project (TMA4320) that utilizes Monte Carlo simulations to study the formation of membrane-less organelles. The project implements numerical methods to model systems of charged monomers and polymers, investigating how these structures evolve over time under various physical conditions.

## Scientific Background

### System Description

The project models a simplified 2-dimensional cellular environment using a discrete grid system. This approach quantifies positions and interactions between monomers while maintaining computational efficiency. The system operates under several key assumptions:

- **Discrete Grid System**: Positions are quantified on a 2D grid representing a plane cross-section of a cell
- **Nearest-Neighbor Interactions**: Only neighboring monomers (distance = unit step length) interact with each other
- **Periodic Boundary Conditions**: The grid behaves as a geometric torus to simulate realistic "outside" boundary effects
- **Electrostatic Interactions**: Energy calculations follow superposition of electric potential principles

### Physical Parameters

The system uses two different scales depending on the type of simulation:

**Monomer System:**

- Unit step length: a = (23 × 10⁻⁶)² m
- Relative permittivity: εᵣ = 78

**Polymer System:**

- Unit step length: a = (91 × 10⁻⁶)² m
- Relative permittivity: εᵣ = 78

## Mathematical Framework

### Energy Model

The total system energy is calculated using electrostatic potential theory:

```
E = Σᵢⱼ Vᵢⱼ
```

where the interaction potential between particles i and j is:

```
Vᵢⱼ = (1/4πε) × (qᵢqⱼ/|rᵢ - rⱼ|)
```

For quantized charges (qᵢ = wᵢe, where wᵢ ∈ Z), this simplifies to:

```
Vᵢⱼ = wᵢwⱼα    (for nearest neighbors)
Vᵢⱼ = 0         (otherwise)
```

where α = e²/(4πε₀εᵣa) is the fundamental energy scale.

### Monte Carlo Algorithm

The simulation employs the Metropolis algorithm with the following acceptance criteria:

1. **Energy Decrease**: Moves that lower system energy are always accepted
2. **Thermal Fluctuations**: Moves that increase energy are accepted with probability:
   ```
   P(accept) = exp(-ΔE/kᴃT)
   ```

This approach ensures convergence toward equilibrium while allowing thermal fluctuations.

## Key Features

### Monomer Simulations

- Generation of random monomer configurations with charges ±1
- Implementation of single-particle Monte Carlo moves
- Energy minimization through electrostatic interactions
- Temperature-dependent clustering behavior analysis

### Polymer Simulations

- Construction of connected polymer chains with length L
- **Rigid Movement**: Entire polymer moves as a single unit
- **Flexible Movement**: Individual segments can move independently while maintaining connectivity
- Cluster formation analysis for different polymer lengths

### Advanced Analysis

- Cluster size distribution calculations
- Temperature dependence of aggregation behavior
- Statistical analysis using mean-of-means estimators
- Equilibration time modeling

## Computational Optimizations

### Numba JIT Compilation

The project extensively uses Numba's Just-In-Time (JIT) compilation to achieve significant performance improvements:

```python
@jit(nopython=True)
def function_name():
    # Optimized function code
```

This optimization enables:

- Execution of millions of Monte Carlo steps
- Large-scale statistical sampling
- Real-time visualization of system evolution

### Performance Considerations

- Efficient neighbor-finding algorithms with periodic boundaries
- Optimized energy calculation routines
- Memory-efficient grid operations
- Parallel-ready algorithm design

## Physical Insights

### Temperature Effects

- **Low Temperature**: Formation of large, stable clusters resembling checkerboard patterns
- **High Temperature**: Increased thermal fluctuations leading to smaller, more dispersed clusters
- **Equilibration Time**: Exponential relationship between temperature and time to reach equilibrium

### Polymer Behavior

- **Rigid vs Flexible**: Flexible polymers achieve lower energy states through shape adaptation
- **Length Dependence**: Relationship between polymer length L and cluster characteristics
- **Contact Optimization**: Polymers with opposite charges maximize contact surfaces

### Scaling Laws

The analysis reveals important scaling relationships:

- Mean cluster size ⟨d⟩ inversely proportional to number of clusters ⟨m⟩
- Product ⟨d⟩⟨m⟩/L = 2M (conservation relationship)
- Temperature-dependent cluster size distributions

## Statistical Methods

### Estimator Selection

The project employs sophisticated statistical methods:

**Mean-of-Means Estimator**:

```
μ̃ = (1/n) Σᵢ₌₁ⁿ (1/mᵢ) Σⱼ₌₁ᵐⁱ xⱼ
```

This approach provides:

- Unbiased estimation of expected values
- Reduced variance compared to simple averaging
- Better precision for cluster size measurements

### Sample Size Determination

Using statistical confidence interval theory:

```
SS = Z²p(1-p)/M²
```

where Z = 1.96 (95% confidence), M = 0.05 (margin of error), leading to n = 385 samples.

## Applications

This computational framework provides insights into:

- **Biological Phase Separation**: Understanding how membrane-less organelles form in cells
- **Self-Assembly Processes**: Mechanisms of spontaneous structure formation
- **Temperature-Dependent Behavior**: How thermal energy affects biological organization
- **Polymer Physics**: Fundamental principles of chain molecule behavior
