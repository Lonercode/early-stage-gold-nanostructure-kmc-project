# Atomistic KMC Modeling of Early-Stage Anisotropic Gold Nanostructure Growth: Temperature-Dependent Facet Kinetics and Symmetry Breaking

## Overview
This repository contains the source code, seed-generation scripts, and raw morphological output data supporting the research manuscript. 

The framework implements a minimal, phenomenological three-phase atomistic Kinetic Monte Carlo (KMC) model incorporating:
- Many-body Embedded-Atom Method (EAM) interactions (Grochola potential via ASE)
- Coordination-number-based Bell-Evans-Polanyi (BEP) surface diffusion energetics
- Phenomenological silver underpotential deposition (UPD) symmetry-breaking models
- Burton-Cabrera-Frank (BCF) edge/kink rate premiums

## Repository Contents
- **`scripts/`**: Python implementation of the Gillespie KMC engine, seed generation, and relaxation routines.
- **`data/`**: Raw morphological output files, including facet populations (N_{111}, N_{100}), geometric aspect ratios (geo_AR), and deposition/diffusion event logs for:
  - Phase 1 (CTAB-only temperature sweeps across 280–340 K, 21 simulations)
  - Phase 2 (Silver UPD symmetry-breaking extensions)
  - Phase 3 (BCF rate premium sweeps and extended volume growth up to 1,500 events)

## Data Availability
All raw simulation outputs and data logs associated with this study are archived within this repository. For specific inquiries regarding the implementation, contact the author.
