# Self-Supervised Learning for EEG-Based Psychiatric Disorder Classification
 Evaluating performance of multiple self-supervised learning methods to utilize unlabeled EEG-data for pretraining for psychiatric disorder classification.
 
 Student: Tuur Smolders
 
 Supervisor: Dr. Marijn van Wingerden

Included in the investigation:
- standard ML models: Random Forest, Support Vector Machine, Gradient Boosting Classifier
- Deep neural network: ShallowNet
- Self-supervised learning tasks: Relative Positioning, Cross-Subject Shuffling, Contrastive Loss
- data: eyes-closed resting-state EEG data from the TDBRAIN dataset.

Analytical pipeline is shown in order of the notebooks. Notebooks containing code for background information can be found in the 'background_notebooks' subdirectory.

See Tuur_Smolders_Tilburg_University_DSS_Masters_Thesis_2024.pdf for the entire report.

Analytical pipeline overview:
![diagram_methodology](https://github.com/user-attachments/assets/e9ec120d-e1a1-4b25-818d-a516a1c6849e)

Boxes represent objects within pipeline. General analytical approaches and direction are represented with solid arrows. Dashed arrows show specific steps within general approaches. Parts of the pipeline that were performed with the TDBRAIN code or the MNE toolbox have been encased in respectively labeled boxes. EEG = electroencephalography; TFR = time-frequency representation; FC = functional connectivity; ML = machine learning; cv = cross-validation.

Contrastive Module pretext model:
![image](https://github.com/user-attachments/assets/00c55b9a-c88f-49d0-9492-34ed57ad4ccd)

Contrastive Loss pretext model:
![image](https://github.com/user-attachments/assets/2d73e552-ef21-4f9d-9a07-e7c178d7a3d2)
