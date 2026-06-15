# Fatigue Analysis Standard

## Scope
This standard governs the fatigue life assessment of metallic structural components
subjected to cyclic loading. It covers stress-life (S-N) and strain-life approaches.

## Stress-Life (S-N) Method
The stress-life method is appropriate for high-cycle fatigue (HCF), where the component
experiences more than approximately 10,000 cycles and stresses remain largely elastic.
- Use the material S-N curve appropriate to the surface finish and loading type.
- Apply a fatigue strength reduction factor (Kf) to account for stress concentrations.
- The endurance limit may be assumed only for steels; aluminium alloys do not exhibit a
  true endurance limit and require a defined design life.

## Strain-Life (E-N) Method
The strain-life method is required for low-cycle fatigue (LCF), where plastic strain is
significant and the life is below approximately 10,000 cycles. Use the Coffin-Manson
relationship with cyclic material properties.

## Mean Stress Correction
Cyclic loading with a non-zero mean stress reduces fatigue life. Apply a mean stress
correction:
- Goodman correction: conservative, recommended for brittle materials and general design.
- Gerber correction: less conservative, suitable for ductile materials.
- Soderberg correction: most conservative, uses yield strength as the limit.

## Fatigue Safety Factor Selection
The fatigue safety factor accounts for scatter in material data, load uncertainty, and
consequence of failure.
- For non-critical components with well-characterised loads: a safety factor of 1.5 to 2.0
  on life or stress is typical.
- For critical components where failure risks safety: use a safety factor of 3.0 or higher.
- When load data is uncertain or estimated, increase the safety factor accordingly.
- Document the basis for the chosen safety factor in the analysis report.

## Cumulative Damage
For variable-amplitude loading, use Miner's rule to sum the damage from each load block.
Failure is predicted when the cumulative damage ratio reaches 1.0. In practice, a damage
ratio limit of 0.5 to 0.7 is often applied as an additional margin for critical parts.

## Weld Fatigue
Welded joints must be assessed using a weld-specific S-N curve (for example, the
structural-stress or nominal-stress approach per IIW recommendations). The weld toe is
typically the critical location. Do not use base-material S-N data for weld assessment.
