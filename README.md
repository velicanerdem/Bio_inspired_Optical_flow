# *Bio-Inspired Computer Vision:  Optical flow estimation using space-time separable filters*

This project is imported from 

### The repository contains the following:  

>-Python code:  folder:src
> > 1.optical_flow.ipynb
> > > a.definitions of  spatial and temporal filters  
> > > b. visualization of filters above in 2d or 3d view   
>>>c.implementation of equation (Eq 23) in [<sup>1</sup>](#refer-anchor-1)[@tschechneBioInspiredOpticFlow2014] and visualization  
>>>d.using aggregation to calculate the velocity at each pixel for optical flow  based on separable filters and visualization(Eq.33 [<sup>1</sup>](#refer-anchor-1))  
>>>e. .npy files are the intermediate files  

>-[Dataset](http://rpg.ifi.uzh.ch/datasets/davis/slider_far.zip):  
>>This is the dataset which is used in this project.For this dataset, only event.txt is used.

>>One can also find other datasets in here: http://rpg.ifi.uzh.ch/davis_data.html

>-Figures:  
>> the generated figures are saved in the folder: output_figures  

>-Slides:
>>This is the folder contains the slides which explains some general ideas of this project.


### Dependencies of this code
numpy,  
matplotlib,  
pandas,  
opencv (cv2),

#### References   

<div id="refer-anchor-1"></div>

- [1] Brosch Tobias, Tschechne Stephan, Neumann Heiko,*[On event-based optical flow detection](https://www.frontiersin.org/article/10.3389/fnins.2015.00137)*

<div id="refer-anchor-2"></div>

- [2] Tschechne, Stephan and Sailer, Roman and Neumann,Heiko.*[Bio-Inspired Optic Flow from Event-Based Neuromorphic Sensor Input](https://link.springer.com/chapter/10.1007/978-3-319-11656-3_16)*  

<div id="refer-anchor-3"></div>

- [3]Tschechne, Stephan and Brosch, Tobias and Sailer, Roman and von Egloffstein, Nora and Abdul-Kreem, Luma Issa and Neumann, Heiko.*[On Event-Based Motion Detection and Integration](https://doi.org/10.4108/icst.bict.2014.257904)*
