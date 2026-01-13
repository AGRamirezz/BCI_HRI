## BCI for HRI

This project sets up a demo to prototype using neurotech for robotic system control. It sought a novel approach to address the following points:

- Encoding and Decoding of Language Dictionaries from Brain Signals for Higher Dimensionality Control Vocabulary 
  - Precursor for LLM based language mapping 
- Multimodal Language Encoding (Many-to-One) for robust language detection 
- Embodied language behavior elicitation for human operators to more seamlessly produce control signals
- Novel signal processing for encoding brain signals into language based latent space representations in deep learning


### Navigating Project Folders:

**Demo-Media-Content**: contains any videos, pictures related to the demo.

**dsi2lsl-app**: this is the app that runs the EEG & makes it available as an LSL stream to access for both the EEG demo 
and the experiment for data collection.

**EEG-Demo**: this contains the sripts to run the EEG-Robot demo that pops up a GUI.

**LabRecorder**: this is used when running the experiment data collection task to record data from both the EEG stream and the experiment
Psychopy stream.

**Run-experiment-data-collection**: this contains the script to run the data collection task. There are also scripts to process the raw xdf files saved by lab recorder into nice & clean numpy arrays. 
The numpy arrays are what is used to pass the EEG data into models.
