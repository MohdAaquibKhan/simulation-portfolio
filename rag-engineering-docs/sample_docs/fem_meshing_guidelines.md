# FEM Meshing Quality Guidelines

## Purpose
This document defines the mesh quality acceptance criteria for structural finite-element
analyses. Meshes that violate these criteria must be refined or remediated before the
analysis is considered valid.

## Element Aspect Ratio
The aspect ratio is the ratio of the longest to the shortest edge of an element.
- Acceptable: aspect ratio below 5 for general structural regions.
- In regions of high stress gradient (notches, fillets, holes), aspect ratio should be
  kept below 3.
- Aspect ratios above 10 are not acceptable and indicate a poorly shaped element that
  will degrade stress accuracy.

## Skewness
Skewness measures how far an element deviates from an ideal shape.
- Excellent: skewness below 0.25.
- Acceptable: skewness below 0.5.
- Poor: skewness between 0.8 and 0.95 — refine these regions.
- Unacceptable: skewness above 0.95, which can cause solver failure.

## Jacobian Ratio
The Jacobian ratio checks for element distortion in mapped/curved elements.
- A Jacobian ratio close to 1.0 indicates a well-formed element.
- Values below 0.6 should be remediated.

## Element Order
- Use second-order (quadratic) elements for stress analysis where bending or stress
  concentrations are present. Quadratic elements capture stress gradients far better than
  linear elements for the same node count.
- Linear elements may be used for contact-dominated or explicit dynamic analyses where
  they are numerically more robust.

## Mesh Refinement and Convergence
A mesh convergence study is mandatory for any stress result used in a design decision.
Refine the mesh until the peak stress in the region of interest changes by less than 5%
between successive refinements. Report the converged element size.

## Minimum Elements Through Thickness
For bending-dominated thin features, use at least 3 second-order elements (or 4 linear
elements) through the thickness to capture the bending stress distribution.
