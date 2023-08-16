# Supplementary data for "Humans-vs.-CNN-Effects-of-task-and-image-type"


## Summary

Do humans and CNN attend to similar areas during scene classification? And how does this depend on the task used to elicit human attention maps? These questions were addressed in an article (link added soon) that compared attention maps generated from human eye movements, human manual selection, or so called e**x**plainable **a**rtificial **i**ntelligence (XAI). The present repository contains the respective source code:

- the CNN architecture (ResNet-152)[^1]
- the XAI method (Grad-CAM)
- the procedures for extracting attention maps from humans and CNN
- statistical evaluation e.g. for calculating dice scores.

If you have any questions regarding the material please contact the corresponding author ([Romy Müller](https://tu-dresden.de/mn/psychologie/iaosp/applied-cognition/die-professur/team/romy-mueller?set_language=en)).

[^1]: Note that, while the architecture of the CNN model consists only of some 100 lines of Python code, the actual weights of the trained model is too big to publish here. Please contact us if you need the trained model for your research.

