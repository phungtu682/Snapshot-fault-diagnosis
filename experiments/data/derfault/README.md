# DER-Fault: A Dataset for Fault Analysis in DER-Integrated Distribution Systems

## Overview

DER-Fault is a simulation-based dataset developed for short-circuit fault analysis in distribution systems with distributed energy resources (DERs). The dataset was generated using the Iowa 240-bus distribution system modeled in OpenDSS. The simulations consider 60 representative operating conditions selected from annual load data and different DER scenarios involving photovoltaic (PV) generation, electric vehicle (EV) integration, and vehicle-to-grid (V2G) operation.

The dataset includes normal operating cases and ten short-circuit fault classes, covering single-line-to-ground, line-to-line, double-line-to-ground, and three-phase-to-ground faults. Pre-fault and post-fault voltage and current measurements were obtained from 93 phasor measurement unit (PMU) locations under different operating conditions, fault locations, and fault parameters.

The dataset is organized with the corresponding fault labels and simulation information, including the operating condition, DER scenario, fault type, and fault location. This allows the data to be used for fault detection, classification, and localization studies, as well as data-driven distribution system monitoring using artificial intelligence (AI) and machine learning (ML) techniques.

A detailed description of the dataset generation procedure, DER modeling, fault simulation, and validation is provided in the associated paper.

## Dataset Summary

| Parameter | Description |
| ---| --- |
| Distribution system 	| 240-bus distribution network |
| Simulation software 	| OpenDSS |
| Representative load conditions  	| 60 |
| PMU equipped buses | 93 |
| Number of classes | 11 (1 normal + 10 fault classes)|
| DERs considered | PV, EV, and V2G |
| Measurements | Voltage and current phasors |
| Data format | CSV |

## Fault Classes

The dataset contains 11 classes, including one normal operating class and ten short-circuit fault classes.

| Label | Class | Description |
| --- | --- | --- |
| 0 | Normal | Normal operating condition |
| 1 | LGA | Phase A-to-ground fault |
| 2 | LGB | Phase B-to-ground fault |
| 3 | LGC | Phase C-to-ground fault |
| 4 | LLAB | Phase A-to-phase B fault |
| 5 | LLBC | Phase B-to-phase C fault |
| 6 | LLAC | Phase A-to-phase C fault |
| 7 | LLABG | Phase A-phase B-to-ground fault |
| 8 | LLBCG | Phase B-phase C-to-ground fault |
| 9 | LLACG | Phase A-phase C-to-ground fault |
| 10 | LLLG | Three-phase-to-ground fault |

## DER Integration 

The dataset considers different levels of DER integration involving
PV generation, EV charging, and
V2G operation.
Six DER scenarios are considered in the dataset:

- **Scenario 1:** No DER penetration.
- **Scenario 2:** 30% PV, 0% EV and 0% V2G.
- **Scenario 3:** 50% PV, 20% EV, and 0% V2G.
- **Scenario 4:** 70% PV, 30% EV, and 0% V2G.
- **Scenario 5:** 70% PV, 30% EV, and 30% V2G.
- **Scenario 6:** 30% PV, 50% EV, and 50% V2G.

## PMU Measurements and Features
Measurements were collected from 93 PMU-equipped buses in the 240-bus
distribution network. The dataset contains pre-fault and post-fault voltage
and current phasor measurements.

The voltage measurements include:
- pre-fault voltage magnitude and phase angle; and
- post-fault voltage magnitude and phase angle.

The current measurements include:

- pre-fault current magnitude and phase angle; and
- post-fault current magnitude and phase angle.
  
PMU locations were determined using the Non-dominated Sorting Genetic
Algorithm II (NSGA-II) by considering the trade-off between the number of
installed PMUs and network observability. PMU placement solutions are
provided for target observability levels of 25%, 50%, 75%, and 100%.

The optimized PMU bus locations for each observability target are provided
in `pmu_placement_configurations.csv`. Each entry identifies the observability target
and the corresponding PMU-equipped bus.

## Dataset Files

The repository contains the following files:

- `DERfaults_measurements.csv` – PMU voltage and current measurements and derived features for the simulated cases.
- `labels_metadata_DERfaults.csv` – fault labels and simulation information associated with each sample.
- `pmu_placement_configurations.csv` – PMU bus locations for the different PMU placement configurations.
- `representative_load_P.csv` – active power (P) values for the representative load conditions used for dataset generation.
- `representative_load_Q.csv` – reactive power (Q) values for the representative load conditions used for dataset generation.
- `README.md` – documentation for the dataset.


## Data Organization & Variables
Each simulated case is assigned a unique `SampleID`. The identifier is used to link the measurement data with the corresponding labels and simulation information.

Each sample represents a specific combination of representative load condition, DER scenario, and fault condition. For fault cases, the associated metadata include the fault type, fault location, and fault resistance. Normal cases represent system operation without an applied short-circuit fault.

### Measurement Data

The measurement file contains `SampleID` together with pre-fault and
post-fault voltage and current phasor measurements.

Voltage columns follow the general format:

`bus<BusID>.<Phase>_<Variable>_<State>`

where:
- `BusID` – distribution system bus number;
- `Phase` – phase number (1, 2, or 3, where applicable);
- `Vmag` – voltage magnitude;
- `Vang` – voltage phase angle;
- `pre` – pre-fault measurement; and
- `post` – post-fault measurement.

Current columns follow the general format:

`Line.<LineID>_T<Terminal>_Ph<Phase>_<Variable>_<State>`

where:
- `LineID` – line identifier;
- `T1` / `T2` – line terminal;
- `Ph1`, `Ph2`, `Ph3` – phase;
- `Imag` – current magnitude;
- `Iang` – current phase angle;
- `pre` – pre-fault measurement; and
- `post` – post-fault measurement.

### Labels and Metadata

The `labels_metadata_DERfaults.csv` file contains the labels and simulation
information associated with each sample.

| Column | Description |
| --- | --- |
| SampleID | Unique identifier for each simulated sample |
| Representative_Load_Index | Index of the representative load condition used for the simulation |
| Timestamp | Timestamp corresponding to the selected representative load condition |
| Hour | Hour associated with the representative load condition |
| Load_Category | Category assigned to the representative load condition |
| Scenario_ID | Identifier of the DER scenario |
| PV_Penetration | PV penetration level (%) |
| EV_Penetration | EV penetration level (%) |
| V2G_Participation | V2G participation level (%) |
| PV_Profile_Mult | PV generation profile multiplier for the corresponding operating condition |
| EV_Charge_Profile_Mult | EV charging profile multiplier for the corresponding operating condition |
| V2G_Profile_Mult | V2G profile multiplier for the corresponding operating condition |
| Fault_Type | Short-circuit fault type or normal operating condition |
| Fault_Label | Numerical class label corresponding to the fault type |
| Fault_Bus | Bus at which the fault is applied |
| Resistance | Fault resistance used in the simulation |

## Software

The distribution system and fault simulations were performed using OpenDSS through the OpenDSSDirect.py interface in Python. Python was also used for simulation automation, data processing, and dataset generation.
## Limitations

The dataset is simulation-based and does not contain field-measured PMU data. The generated measurements depend on the distribution network model, load conditions, DER models, PMU configuration, and fault parameters considered in the simulations.
The 60 representative load conditions were selected to represent variations in the annual load data and do not include every hourly operating condition. The DER and fault scenarios are also limited to the configurations considered during dataset generation.

## Source Data 

The Iowa 240-bus distribution system model and annual load data used to construct the simulation environment were obtained from [1]. 
The original distribution network model and annual load data are available at: https://wzy.ece.iastate.edu/Testsystem.html.
These source materials are not redistributed in this repository.

The dataset deposited in this repository consists of the simulation outputs and supporting information generated in this study. Further details on the source data, DER profiles, and dataset generation
procedure are provided in the associated paper.

#### References

[1] F. Bu, Y. Yuan, Z. Wang, K. Dehghanpour, and A. Kimber, "A Time-series Distribution Test System based on Real Utility Data." 2019 North American Power Symposium (NAPS), Wichita, KS, USA, 2019, pp. 1-6. 

## Associated Paper

A detailed description of the dataset generation procedure, simulation setup, DER modeling, fault scenarios, and validation is provided in the associated paper:  
 F.R.M. Razick, P. Musilek, "DER-Fault: A Simulation-Based Fault Dataset for DER-Integrated Distribution Systems Under Diverse Operating Conditions," arXiv preprint, 2026. (To be made available on arXiv.)

For publications referring to the dataset generation methodology, please cite the associated paper.

## License

This dataset is released under the Creative Commons Attribution 4.0
International (CC BY 4.0) license. This license applies only to the materials generated and released by the authors.

## Funding
This research has been supported by the Natural Sciences and Engineering Research Council (NSERC) of Canada grant number ALLRP 549804–19, and by AESO, Altalink, ATCO, Enmax, EPCOR, and FortisAlberta.



