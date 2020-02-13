# OpenVINO_SmartCam

The aim of the project is to create an AI-powered Smart Cameras that can mitigate the spread of Wuhan Novel Coronavirus.

An AI powered Smart Camera can improve the current thermal scanning checkpoints, that function as nothing more than visual thermometers, in a number of different ways:

- **Unmanned Thermal Imaging**: current thermal scanning checkpoints need to be manned posing a staffing challenge to keep these terminals running 24/7. The screening procedure can be automated by creating models able to automatically diagnose fever.
- **Detect additional symptoms**: AI techniques can be used for detecting additional symptoms, like behavioral ‘events’ such as someone *sneezing* or *coughing*.
- **Potential spread tracking**: AI techniques can be used for keep track of contact between peoples.


The current focus of the project is on the last two point, mainly because is not involved tharmal scanning and the techniques required were in part seen during the Intel Edge AI challenge course.

## Symptoms detection

The pipeline for detecting sneezing or couching involve the use of pose estimation models and the creation of an additional post-processing procedure.


## Human contact tracking

The pipeline for detecting human contact involve the use of human detection models and the creation of an additional post-processing procedure.

# Quickstart

## Prerequisites

You must first install the following libraries:

```txt
...
```

# References

1. [How smart thermal cameras can mitigate the spread of the Wuhan Coronavirus](https://anyconnect.com/blog/smart-thermal-cameras-wuhan-coronavirus)
2. [Sneeze dataset](https://research.google.com/audioset/balanced_train/sneeze.html)
3. [Recognizing flu-like symptoms from videos](https://www.researchgate.net/publication/265607317_Recognizing_flu-like_symptoms_from_videos)
4. [Action Recognition Datasets: "NTU RGB+D" Dataset and "NTU RGB+D 120" Dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp)
