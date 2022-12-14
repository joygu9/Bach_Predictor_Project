README for Project Code
Author: Joy Gu

=== Environment setup ===

The transfer learning code uses ADAPT Awesome Domain Adaptation Python Toolbox
https://adapt-python.github.io/adapt/index.html

My system is an M2 chip MacBook Air, so I needed to install tensorflow differently because the pip install of ADAPT did not work for me.
I copied the adapt source code into the project folder because the pip install wasn't working.

I followed this tutorial to install tensor flow onto my MacBook:
https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-conda-mac-metal-jul-2022.ipynb
Corresponding video: https://www.youtube.com/watch?v=5DgWvU0p2bk

To run the code, make sure you are on a tensorflow environment, I also needed to install cvxopt as I did not have that before and it's a requirement for ADAPT.


Once environment is configured, you can call the main function from the code directory: python3 main.py

main.ipynb is the same as main.py, but provided as well in case you want to run through Jupyter Notebook