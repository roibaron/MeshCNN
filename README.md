# Modifiying MeshCNN

## The assignment
We were required to apply some modifications on the original <a href="https://ranahanocka.github.io/MeshCNN/">MeshCNN</a>
 project.
The task was to achieve better accuracy results on 2 specific data sets:
<ul>
<li>Human (segmentation task)</li>
<li>Cubes (classification task)</li>
</ul>
For details about the project, citing and howtos, please visit the  <a href="https://github.com/ranahanocka/MeshCNN/">main repo</a>.

### NOTE
This isn't a good practice for improving a DL model. All we achieved is tighter fitting to specific datasets, which probably results in OVERFIT and worse results on general data.
It's not enough that we separate data sets to train and test sets, because we can't measure what matters – how the algorithm will perform on new data.
A better practice would be splitting the data into 3 sets: train, validation and test, or to use cross validation by splitting the data into folds.
The validation set assigned to setting and optimizing the Hyperparameters while the test set to estimate the final model success, and hence the model can't reach this data before it's done.

# What we tried?
<ol class="c16 lst-kix_wfe3e81sw3y3-0 start" start="1"><li class="c6"><span class="c3">Activation function we tried using tanh, sigmoid and leaky relu instead of the relu layers.</span></li><li class="c6"><span class="c3">Modifying learning rate</span></li><li class="c6"><span class="c3">Tweaking the batch size</span></li><li class="c6"><span class="c3">Changing </span></li><li class="c6"><span class="c3">Changing the loss reduction type from mean to std and median.</span></li><li class="c6"><span class="c3">Changing init type</span></li><li class="c6"><span class="c3">Modifying resblock</span></li><li class="c6"><span class="c3">Adding kernel features: Norm / Sum /Std /Multiplication and division of the edges (which resulted in NaN)</span></li><li class="c6"><span class="c3">Adding dropout - during training we zero some of the activations (and don’t update those weights). It is a kind of regularization- good for generalization. </span></li><li class="c6"><span class="c3">Adding layers to the classification network. We tried adding a fully connected layer between any 2 convolution layers.</span></li><li class="c6"><span class="c3">Changing the parameter for edge collapse: Randomly collapsing edges</span></li></ol>

We also used many combinations of the above
Partial results and code for these modifications is attached.


## The final model
 The final model included just modest tweaks:
Replacing the activation function.
Adding a sum feature to the kernal.

# The results


<table class="c33"><tbody><tr class="c8"><td class="c11" colspan="2" rowspan="1"><p class="c1"><span class="c3">segmentation</span></p></td><td class="c13" colspan="2" rowspan="1"><p class="c1"><span class="c16 c37">classification</span></p></td></tr><tr class="c28"><td class="c14" colspan="1" rowspan="1"><p class="c1"><span class="c16">1</span><span class="c16 c27">st</span><span class="c3">&nbsp; run</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c1"><span class="c16">2</span><span class="c16 c27">nd</span><span class="c3">&nbsp;run</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c16">1</span><span class="c16 c27">st</span><span class="c3">&nbsp; run</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c16">2</span><span class="c16 c27">nd</span><span class="c3">&nbsp;run</span></p></td></tr><tr class="c8"><td class="c14" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;95.094%</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;95.669%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;97.56%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;</span></p></td></tr><tr class="c8"><td class="c14" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;95.085%</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;95.203%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;97.37%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;</span></p></td></tr><tr class="c8"><td class="c14" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;95.106%</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;94.851%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;96.47%</span></p></td><td class="c19" colspan="1" rowspan="1"><p class="c1"><span class="c0">&nbsp;</span></p></td></tr><tr class="c8"><td class="c11" colspan="2" rowspan="1"><p class="c1"><span class="c30">Average</span><span class="c0">&nbsp;95.168</span></p></td><td class="c13" colspan="2" rowspan="1"><p class="c1"><span class="c25">average</span></p></td></tr><tr class="c8"><td class="c11" colspan="2" rowspan="1"><p class="c1 c23"><span class="c25"></span></p></td><td class="c13" colspan="2" rowspan="1"><p class="c1 c23"><span class="c25"><img src='docs/imgs/cubes accuracy.jpeg' align="right" width=325>
</span></p></td></tr></tbody></table>
