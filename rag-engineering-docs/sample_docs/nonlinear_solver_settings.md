# Nonlinear Solver Settings — Tool Documentation

## Overview
This guide documents the solver controls for nonlinear static structural analyses,
covering convergence behaviour for contact, large-deflection, and material nonlinearity.

## Large Deflection
Enable large-deflection effects when displacements or rotations are large enough that
the structure's stiffness changes with deformation (geometric nonlinearity). Examples:
snap-fits, thin shells under buckling, and slender members. Leaving large deflection off
in these cases produces non-physical, overly stiff results.

## Substeps and Time Stepping
Nonlinear problems are solved incrementally over substeps.
- Increase the number of substeps if the solution fails to converge — smaller load
  increments are easier to converge.
- Enable automatic time stepping so the solver can bisect the load increment when it
  encounters difficulty and grow it again when convergence is easy.
- Set a sensible minimum substep so the solver does not stall at an excessively small step.

## Contact Settings
Contact is the most common source of convergence trouble.
- Use the Augmented Lagrange formulation as a robust default for most contact.
- Reduce the normal contact stiffness factor (FKN) if the solver reports large
  penetration oscillations or fails to converge; a value of 0.1 to 1.0 is typical.
- Add a small amount of contact stabilisation damping to overcome rigid-body motion
  before contact is established.
- Check for initial gaps or interference; use contact offset or "adjust to touch" to
  close small modelling gaps that would otherwise cause rigid-body motion.

## Convergence Criteria
- The default force convergence tolerance is 0.5% of the reference force. Tightening it
  improves accuracy but costs iterations.
- If a model converges on force but not displacement, enable displacement convergence
  checking as well.
- Watch the Newton-Raphson residual plots: residuals concentrated at a single node
  usually indicate a local mesh or contact problem there.

## Stabilisation for Unstable Problems
For problems with local instabilities (buckling, snap-through), enable nonlinear
stabilisation with an energy-dissipation ratio. Start with a small dissipation value and
verify that the stabilisation energy remains a small fraction of the total strain energy,
so the added damping does not corrupt the result.

## Solver Choice
- The sparse direct solver is robust and the default for most structural models.
- The iterative PCG solver can be faster and use less memory for very large, well-
  conditioned models, but may struggle with ill-conditioned contact problems.
