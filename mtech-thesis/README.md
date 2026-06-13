# M.Tech Thesis — Thermoplastic Failure Characterisation

**Degree:** M.Tech. in Design Engineering  
**Institution:** BITS Pilani  
**Year:** 2021–2023  
**Grade:** CGPA 9.83 / 10 (Batch Topper)

---

## Thesis Title

**MATLAB Application for Failure Characterisation of Thermoplastic Polymers using the Dissipated Energy Method**

---

## Problem Statement

Thermoplastic polymers are increasingly used in structural components (housings, brackets, clips, snap-fits) in consumer appliances and automotive parts. Predicting when they fail under combined mechanical and thermal loading requires models that capture both:
- **Viscoelastic behaviour** — time-dependent deformation under sustained load (creep)
- **Viscoplastic behaviour** — permanent deformation beyond the elastic limit

Traditional stress-based failure criteria are inaccurate for these materials because failure depends on the loading history, not just the instantaneous stress.

---

## Approach

The **dissipated energy method** treats failure as a thermodynamic event: failure initiates when the cumulative energy dissipated per unit volume reaches a critical threshold, regardless of loading rate or stress state.

### Steps implemented in the MATLAB application:

1. **Data import** — Load DMA (Dynamic Mechanical Analysis) and creep/recovery test data from CSV
2. **Viscoelastic model fitting** — Fit a Generalised Maxwell (Prony series) model to storage/loss modulus data
3. **Viscoplastic parameter extraction** — Extract Perzyna-type viscoplastic parameters from creep/recovery curves
4. **Dissipated energy calculation** — Numerically integrate the stress-strain work over arbitrary loading histories
5. **Failure prediction** — Compare accumulated dissipated energy to the critical threshold
6. **Validation** — Predict failure for independent loading conditions not used in parameter fitting

### MATLAB application features:
- GUI interface for loading test data and setting model parameters
- Automated curve-fitting using `fminsearch` / `lsqcurvefit`
- Numerical time integration of constitutive equations
- Failure prediction plots: dissipated energy vs. time, with failure threshold overlay
- Export of fitted material parameters to a structured format

---

## Key Results

- Model predictions matched independent experimental failure points within ±8% for the tested polymer grades
- Application successfully characterised three thermoplastic grades: POM, PA66-GF30, PP-TD20
- Material parameter tables generated for direct use in Ansys Mechanical nonlinear analyses

---

## Relevance to Industry Work

This thesis directly informed the temperature-dependent material modelling work done at BSH Hausgeräte GmbH (Bosch Group), where creep and plasticity models built from physical test data are used in production FEA workflows.

---

*Note: The full MATLAB source code is proprietary to BITS Pilani. This README describes the methodology and outcomes.*
