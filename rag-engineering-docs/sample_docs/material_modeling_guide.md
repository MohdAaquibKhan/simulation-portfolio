# Material Modeling Selection Guide

## Purpose
Guidance on selecting the appropriate constitutive material model for a structural
simulation, based on the material and the loading regime.

## Linear Elastic
Use a linear elastic model when stresses remain below the yield strength everywhere and
the analysis is concerned only with elastic response. Requires Young's modulus and
Poisson's ratio. This is the fastest and most robust model.

## Elastic-Plastic (Metal Plasticity)
Use when stresses exceed yield and permanent deformation must be captured.
- Bilinear isotropic hardening: simplest plastic model, needs yield strength and a
  tangent modulus. Good for monotonic loading.
- Multilinear isotropic hardening: defines the true stress-strain curve as points;
  recommended when the hardening behaviour is strongly nonlinear.
- Kinematic hardening: required for cyclic loading where the Bauschinger effect and
  reversed plasticity matter, such as low-cycle fatigue.
- Always input the TRUE stress-strain curve, not engineering stress-strain, for plastic
  models.

## Hyperelastic (Rubber and Elastomers)
Use for rubber-like materials that undergo large recoverable strains.
- Mooney-Rivlin and Ogden are common models; fit their coefficients to uniaxial,
  biaxial, and planar test data.
- A single uniaxial test is usually insufficient to characterise a hyperelastic material
  reliably; obtain multiple test modes where possible.

## Viscoelastic and Viscoplastic
Use when the material response depends on time and loading rate, such as thermoplastics
and polymers under sustained load.
- Viscoelastic models (Prony series) capture creep and stress relaxation.
- Viscoplastic models capture rate-dependent permanent deformation.
- Characterise these models from DMA (dynamic mechanical analysis) and creep/recovery
  test data across the relevant temperature range.

## Creep
Use a creep model for components held at elevated temperature under sustained load over
long times, where time-dependent deformation accumulates. Creep parameters are
temperature-dependent and must be fitted from creep test data at the service temperature.

## Temperature-Dependent Properties
When a component sees a significant temperature range, define material properties as
functions of temperature. Stiffness, yield, and thermal expansion all vary with
temperature, and ignoring this in a thermo-structural analysis can substantially
mispredict stresses.
