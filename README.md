# xVAEnet
 
## Data
Each trial in the dataset is composed of 23 channels and 3001 timestamps, as shown on Figure 1
![alt text](https://github.com/numediart/xVAEnet/blob/main/data.png)
Fig. 1. Data overview. Example of a preprocessed 60-seconds trial with OSA
event. Channels: 1) nasal airflow, 2-3) abdominal-thoracic respiratory motions,
4) oxygen saturation, 5-6) electrooculograms, 7) pulse rate variability, 8)
abdominal-thoracic motions phase shift, 9-23) EEG signal of the 3 selected
electrodes at different frequency ranges.

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

## Architecture
![alt text](https://github.com/numediart/xVAEnet/blob/main/detailed_architecture.png)
Fig2. Detailed xVAEnet architecture

## Training
The entire training phase has been
performed on an NVIDIA GeForce GTX 1080Ti 12Go RAM
on 12 workers.
The first module to be trained is the VAE module. The training
process has been performed on the trainset on 100 epochs with
a batch size of 16, a learning rate of 5 · 10−3 and a ranger
optimizer, while the validation has been done on the testset
with a batch size of 32. A gradient accumulation of 64 samples
and an early stopping option based on the validation loss with
a patience of 10 epochs have also been used.
Then, the GAN module has been trained by initializing the
generator with the best weights of the encoder obtained during
the VAE training phase and the discriminator is randomly
initialized. At each batch, the discriminator is first trained
by freezing the generator and using the loss function of the
discriminator described in Section III, then the generator is
trained by freezing the discriminator and using the correspond-
ing loss function. This training phase is performed on the
trainset on 100 epochs with a batch size of 16, a learning rate
of 2 · 10−3 and a root mean square propagation (RMSprop)
optimizer, while the validation process is done with a batch
size of 32. A gradient accumulation of 64 samples and an early
stopping option with a patience of 30 are also used. Every 15
epochs, the updated network is used in inference to compute a
new Zd vector given as real input for the 15 following epochs
in order to avoid the deterioration of the “real” space to be
responsible for the increase of the GAN performance.
Finally, the classifier module is initialized with the weights
of the best generator previously obtained and the single-
layer perceptron is randomly initialized. In the philosophy of
curriculum learning, the classifier is trained on each severity
feature sequentially, starting with the low vs. high severity
classification on the hypoxic burden, then on the arousal index
and finally on the event duration. All the training processes
for the classification task have been performed on the trainset
on 100 epochs with a batch size of 16 and a ranger optimizer,
and have been validate on the testset with a batch size of 32
only the learning rate and the combination of loss functions
differ. A gradient accumulation of 64 samples and an early
stopping option based on the validation loss with a patience
of 30 epochs have also been used. On the hypoxic burden, the
learning rate was set to 10−3 and, once every 5 epochs, the
whole model was trained using a global loss combining the
three modules, as described in Equation 1.
```math
\mathcal{L}_{global} = \frac{1}{3}\mathcal{L}_{VAE} + \frac{1}{3} \frac{1}{bs} \sum_{i=1}^{bs} (1-fake\_pred_{i}) + \frac{1}{3}\mathcal{L}_{classif}
```
