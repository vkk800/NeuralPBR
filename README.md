# NeuralPBR
Generates physically based rendering (PBR) texture maps from images using neural networks.

This is my experimental project to automate the generation of textures needed for physically
based rendering in computer graphics using neural networks.

The project is at its infancy and does not produce very good results yet. A big problem at the
moment is that training the models requires a beefy GPU; the textures need to be big
to create results that are useful (I'm trying to use 2048x2048 size if possible) and processing
these images required a lot of GPU memory. I have some experimental ideas to overcome this (see for example
cropped U-Net model included in the source).

TODO list for the near future is (this will hopefully bring the project to such a state that it's possible for
other people to try it out and hopefully contribute):
* Upload the training datasets somewhere (they come from cc0textures.com, but I have preprocessed them a bit)
* Upload some preliminary example results
* Make a real plan and roadmap/todo-list on how to improve the results
