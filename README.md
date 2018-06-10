# NeuralPBR
Generates physically based rendering (PBR) texture maps from images using neural networks.

This is my experimental project to automate the generation of textures needed for physically
based rendering in computer graphics using neural networks.

The project is at its infancy and does not produce very good results yet. A big problem at the
moment is that training the models requires a beefy GPU; the textures need to be big
to create results that are useful (I'm trying to use 2048x2048 size if possible) and processing
these images required a lot of GPU memory. I have some experimental ideas to overcome this (see for example
cropped U-Net model included in the source).

At the moment I'm mainly trying out different architectures to find out which one should be used
in the end. The status of this work so far for different models is:

* Simple convolutional filter stack: works, but doesn't produce good results (which is expected). Learns basically some kind of black-and-white filter when used to produce displacement maps.
* Dilated version of above: The same. Might be better if more layers and larger dilation is used.
* U-Net: Seemed promising, but was too heavy for my computer using reasonable image resolutions.
* Cropped U-Net: Not tested. I will try it when I get access to better computational resources.
* Autoencoder: Did not work well in the initial testing. Should try different variations of the model and with longer training times on better hardware.

TODO list for the near future is (this will hopefully bring the project to such a state that it's possible for
other people to try it out and hopefully contribute):
* Upload the training datasets somewhere (they come from cc0textures.com, but I have preprocessed them a bit)
* Upload some preliminary example results
* Make a real plan and roadmap/todo-list on how to improve the results
