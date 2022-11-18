# xVAEnet
 
## Preprocessing
The EEG signals have been preprocessed
following the COBIDAS MEEG recommendations from the
Organization for Human Brain Mapping (OHBM) [1]. Trials
significantly affected by ocular artifacts have been excluded
from the database, based on the correlation between the EOG
and the FP1 signals. Trials with non-physiological amplitudes
are also excluded, based on their peak-to-peak voltage (VPP):
VP-P < 10−7V and VP-P > 6 ∗ 10−4V are excluded. A
baseline correction was applied using a segment of 10 seconds
preceding each trial as the baseline. The EEG delta band power
being the most varying frequency band during sleep apneahypopnea
occurrence [2], we focused our analysis on low
frequency EEG components by filtering the signals into 2Hz
narrow bands: 0-2Hz, 2-4Hz, 4-6Hz, 6-8Hz, and 8-10Hz. We
also rejected trials based on physiological fixed range criteria
on VP-P for EOG and SAO2 signals, moreover trials with
VAB, VTH and NAF2P statistical outliers in amplitude are
rejected. Two additional signals have been computed from the
aforementioned recorded signals: 1) the Pulse Rate Variability
(PRV) being the difference between a PR sample and the
next one, and 2) the belts phase shift (Pshift), computed as
the sample by sample phase difference between VAB and
VTH phase signals, as suggested by Varady et al. [3]. The
normalization has been performed by channel independently
as a z-score normalization with clamping in the [-3; 3] range.
After the exclusion and preprocessing phases, the final dataset
is composed of 6992 OSA trials from 60 patients divided
into a training set of 4660 trials from 48 patients, namely
the trainset, and a validation set of 2332 trials from the 12
remaining patients, namely the testset.

[1] C. Pernet, M. I. Garrido, A. Gramfort, N. Maurits, C. M. Michel,
E. Pang, R. Salmelin, J. M. Schoffelen, P. A. Valdes-Sosa, and A. Puce,
“Issues and recommendations from the OHBM COBIDAS MEEG committee
for reproducible EEG and MEG research,” Nature Neuroscience,
vol. 23, no. 12, pp. 1473–1483, Dec. 2020, number: 12 Publisher: Nature
Publishing Group.

[2] C. Shahnaz, A. T. Minhaz, and S. T. Ahamed, “Sub-frame based apnea
detection exploiting delta band power ratio extracted from EEG signals,”
in 2016 IEEE Region 10 Conference (TENCON), Nov. 2016, pp. 190–
193, iSSN: 2159-3450.

[3] P. Varady, S. Bongar, and Z. Benyo, “Detection of airway obstructions
and sleep apnea by analyzing the phase relation of respiration movement
signals,” IEEE Transactions on Instrumentation and Measurement,
vol. 52, no. 1, pp. 2–6, Feb. 2003, conference Name: IEEE Transactions
on Instrumentation and Measurement.