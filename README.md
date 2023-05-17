# MaMaDroid2.0


This repository follows the paper "Breaking the Structure of MaMaDroid".
It contains extended version of the results of the experiments on MaMaDroid.

## Results
There are three results' folders - Detection rate, DRR and Model Reliability.
Each folder depict a different case, which correlates to a different sub-folder: 
1. Original apps
2. Random STB attack
3. Full Statisticl STB attack
4. Black Hole STB attack

Each sub-folder contains 5 CV of the results of the original models and non-mentioned-in-the-original-paper models. The originals are depicted as _former_algs_CVnumber and the new ones only by the CVnumber.

## MaMaDroid2.0
The attached scripts run the MaMaDroid2.0 detection machine against the StB and MB attacks.

## Attack
The attack folder includes the template for the StB attack.

## Cite
If you use this code please cite the following paper:

@article{berger2023breaking,
  title={Breaking the structure of MaMaDroid},
  author={Berger, Harel and Dvir, Amit and Mariconti, Enrico and Hajaj, Chen},
  journal={Expert Systems with Applications},
  pages={120429},
  year={2023},
  publisher={Elsevier}
}
