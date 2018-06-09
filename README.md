# NeuralPBR
Generates physically based rendering (PBR) texture maps from images using neural networks.

This is my experimental project to automate the generation of textures needed for physically
based rendering in computer graphics using neural networks.

The project is at its infancy and does not produce very good results yet. A big problem at the
moment is that I do not have good enough GPU to train the models; the textures need to be rather big
to create results that are useful to people (I'm trying to use 2048x2048 size if possible) and processing
these images required a lot of GPU memory. I have some experimental ideas to overcome this (see for example
cropped U-Net model included in the source).

In the near future, I'm going to upload enough of the project here to allow other people to try it out themselves.
At the moment the training data is missing (I have find a place to upload it). Also I will write some more
documentation and upload some low quality preliminary results.
