# xVAEnet
This repository provides the open-source codes and supplementary materials related to the publication: \\
La Fisca \textit{et al.}, "Explainable AI for EEG Biomarkers Identification
in Obstructive Sleep Apnea Severity Scoring Task", 2023.
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
three modules:
```math
\mathcal{L}_{global} = \frac{1}{3}\mathcal{L}_{VAE} + \frac{1}{3} \frac{1}{bs} \sum_{i=1}^{bs} (1-fake\_pred_{i}) + \frac{1}{3}\mathcal{L}_{classif}
```
On the arousal index, the learning rate was set to 5 · 10−4, the
global training was performed every 5 epochs and, once every
2 epochs, the classification has been performed on both the
hypoxic burden and the arousal index using a weighted sum
of both losses: 
```math
\mathcal{L}_{classif_2} = \frac{1}{2}\cdot \mathcal{L}_{hypox} + \frac{1}{2}\cdot \mathcal{L}_{arousal}
```
On the
respiratory event duration, the learning rate was set to 2·10−4,
the global training was performed every 5 epochs and, twice
every 3 epochs, the classification has been performed on all the
severity features+ using a weighted sum of all classification
losses: 
```math
\mathcal{L}_{classif_3} = \frac{1}{3}\cdot \mathcal{L}_{hypox} + \frac{1}{3}\cdot \mathcal{L}_{arousal} + \frac{1}{3}\cdot \mathcal{L}_{duration}
```

## References
[1] A. S. Jordan, D. G. McSharry, and A. Malhotra, “Adult obstructive sleep
apnoea,” The Lancet, vol. 383, no. 9918, pp. 736–747, Feb. 2014.\\
[2] R. B. Berry, R. Budhiraja, D. J. Gottlieb, D. Gozal, C. Iber, V. K.
Kapur, C. L. Marcus, R. Mehra, S. Parthasarathy, S. F. Quan, S. Redline,
K. P. Strohl, S. L. D. Ward, and M. M. Tangredi, “Rules for Scoring
Respiratory Events in Sleep: Update of the 2007 AASM Manual for
the Scoring of Sleep and Associated Events,” Journal of Clinical Sleep
Medicine, vol. 08, no. 05, pp. 597–619, 2012, publisher: American
Academy of Sleep Medicine.
[3] D. A. Pevernagie, B. Gnidovec-Strazisar, L. Grote, R. Heinzer, W. T.
McNicholas, T. Penzel, W. Randerath, S. Schiza, J. Verbraecken, and
E. S. Arnardottir, “On the rise and fall of the apneahypopnea index:
A historical review and critical appraisal,” Journal of Sleep Research,
vol. 29, no. 4, p. e13066, 2020.
[4] A. Malhotra, I. Ayappa, N. Ayas, N. Collop, D. Kirsch, N. Mcardle,
R. Mehra, A. I. Pack, N. Punjabi, D. P. White, and D. J. Gottlieb,
“Metrics of sleep apnea severity: beyond the apnea-hypopnea index,”
Sleep, vol. 44, no. 7, p. zsab030, Jul. 2021.
[5] D.-H. Park, C.-J. Shin, S.-C. Hong, J. Yu, S.-H. Ryu, E.-J. Kim, H.-B.
Shin, and B.-H. Shin, “Correlation between the Severity of Obstructive
Sleep Apnea and Heart Rate Variability Indices,” Journal of Korean
Medical Science, vol. 23, no. 2, p. 226, 2008.
[6] G. Bachar, B. Nageris, R. Feinmesser, T. Hadar, E. Yaniv, T. Shpitzer,
and L. Eidelman, “Novel grading system for quantifying upper-airway
obstruction on sleep endoscopy,” Lung, vol. 190, no. 3, pp. 313–318,
Jun. 2012.
[7] A. Kulkas, P. Tiihonen, P. Julkunen, E. Mervaala, and J. T ̈oyr ̈as, “Novel
parameters indicate significant differences in severity of obstructive sleep
apnea with patients having similar apnea–hypopnea index,” Medical &
Biological Engineering & Computing, vol. 51, no. 6, pp. 697–708, Jun.
2013.
[8] A. Muraja-Murro, A. Kulkas, M. Hiltunen, S. Kupari, T. Hukkanen,
P. Tiihonen, E. Mervaala, and J. T ̈oyr ̈as, “Adjustment of apnea-hypopnea
index with severity of obstruction events enhances detection of sleep
apnea patients with the highest risk of severe health consequences,”
Sleep and Breathing, vol. 18, no. 3, pp. 641–647, Sep. 2014.
[9] H. Korkalainen, J. T ̈oyr ̈as, S. Nikkonen, and T. Lepp ̈anen, “Mortality-
risk-based apnea–hypopnea index thresholds for diagnostics of obstruc-
tive sleep apnea,” Journal of Sleep Research, vol. 28, no. 6, p. e12855,
2019, eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/jsr.12855.
[10] W. Cao, J. Luo, and Y. Xiao, “A Review of Current Tools Used for
Evaluating the Severity of Obstructive Sleep Apnea,” Nature and Science
of Sleep, vol. 12, pp. 1023–1031, Nov. 2020, publisher: Dove Press.
[11] A. Zinchuk and H. K. Yaggi, “Phenotypic Subtypes of OSA: A Chal-
lenge and Opportunity for Precision Medicine,” CHEST, vol. 157, no. 2,
pp. 403–420, Feb. 2020, publisher: Elsevier.
[12] G. Labarca, J. Gower, L. Lamperti, J. Dreyse, and J. Jorquera, “Chronic
intermittent hypoxia in obstructive sleep apnea: a narrative review from
pathophysiological pathways to a precision clinical approach,” Sleep and
Breathing, vol. 24, no. 2, pp. 751–760, Jun. 2020.
[13] S. Pusk ́as, N. Koz ́ak, D. Sulina, L. Csiba, and M. T. Magyar, “Quantita-
tive EEG in obstructive sleep apnea syndrome: a review of the literature,”
Reviews in the Neurosciences, vol. 28, no. 3, pp. 265–270, Apr. 2017.
[14] E. Sforza, S. Grandin, C. Jouny, T. Rochat, and V. Ibanez, “Is waking
electroencephalographic activity a predictor of daytime sleepiness in
sleep-related breathing disorders?” The European Respiratory Journal,
vol. 19, no. 4, pp. 645–652, Apr. 2002.
[15] K. Dingli, T. Assimakopoulos, I. Fietze, C. Witt, P. K. Wraith, and
N. J. Douglas, “Electroencephalographic spectral analysis: detection
of cortical activity changes in sleep apnoea patients,” The European
Respiratory Journal, vol. 20, no. 5, pp. 1246–1253, Nov. 2002.
[16] M. Younes, A. Azarbarzin, M. Reid, D. R. Mazzotti, and S. Redline,
“Characteristics and reproducibility of novel sleep EEG biomarkers and
their variation with sleep apnea and insomnia in a large community-
based cohort,” Sleep, vol. 44, no. 10, p. zsab145, Oct. 2021.
[17] S. Nikkonen, H. Korkalainen, S. Kainulainen, S. Myllymaa, A. Leino,
L. Kalevo, A. Oksenberg, T. Lepp ̈anen, and J. T ̈oyr ̈as, “Estimating
daytime sleepiness with previous night electroencephalography, elec-
trooculography, and electromyography spectrograms in patients with
suspected sleep apnea using a convolutional neural network,” Sleep,
vol. 43, no. 12, p. zsaa106, Dec. 2020.
[18] G. C. Guti ́errez-Tobal, D.  ́Alvarez, A. Crespo, F. del Campo, and
R. Hornero, “Evaluation of Machine-Learning Approaches to Estimate
Sleep Apnea Severity From At-Home Oximetry Recordings,” IEEE
Journal of Biomedical and Health Informatics, vol. 23, no. 2, pp. 882–
892, Mar. 2019, conference Name: IEEE Journal of Biomedical and
Health Informatics.
[19] M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, “The false hope of
current approaches to explainable artificial intelligence in health care,”
The Lancet Digital Health, vol. 3, no. 11, pp. e745–e750, Nov. 2021.
[20] P. Angelov and E. Soares, “Towards explainable deep neural networks
(xDNN),” Neural Networks, vol. 130, pp. 185–194, Oct. 2020.
[21] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier,
C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, and
M. H ̈am ̈al ̈ainen, “MEG and EEG data analysis with MNE-Python,”
Frontiers in Neuroscience, vol. 7, 2013.
[22] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” May
2014, arXiv:1312.6114 [cs, stat].
[23] X. Zhang, L. Yao, and F. Yuan, “Adversarial Variational Embedding
for Robust Semi-supervised Learning,” in Proceedings of the 25th ACM
SIGKDD International Conference on Knowledge Discovery & Data
Mining, ser. KDD ’19. New York, NY, USA: Association for Computing
Machinery, Jul. 2019, pp. 139–147.
[24] N. A. Eiseman, M. B. Westover, J. M. Ellenbogen, and M. T. Bianchi,
“The Impact of Body Posture and Sleep Stages on Sleep Apnea Severity
in Adults,” Journal of Clinical Sleep Medicine, vol. 08, no. 06, pp. 655–
666, 2012, publisher: American Academy of Sleep Medicine.
[25] S. G. Jones, B. A. Riedner, R. F. Smith, F. Ferrarelli, G. Tononi,
R. J. Davidson, and R. M. Benca, “Regional Reductions in Sleep
Electroencephalography Power in Obstructive Sleep Apnea: A High-
Density EEG Study,” Sleep, vol. 37, no. 2, pp. 399–407, Feb. 2014.