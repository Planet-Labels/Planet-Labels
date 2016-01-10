# Planet-Labels

## Introduction
We are using machine learning and computer vision techniques to classify land cover from satellite images we gathered from the San Francisco start-up Planet Labs. With this project, we can differentiate between a large number of land cover classes, such as forest, crops, barren land, water, wetlands etc.

For example, we could have images like ![Original image](/img/original.jpg) which could be broken down into tiles, for which labels could be predicted, leading us to ![Predicted image](/img/predicted.jpg)
The differently coloured squares represent different classes of land cover in the above picture. 

This could be helpful to environmentalists who want to know of illegal deforestation and the shrinking rate of certain water bodies, and also to urban planners to track the growth of cities. To aid this, we are working on including support for temporal applications of satellite data. 

## Data
We got our satellite image data from [Planet Labs Explorer program](https://www.planet.com/explorers/). For true labels to train supervised machine learning models, we used the [National Land Cover Dataset (NLCD)](http://www.mrlc.gov/nlcd01_data.php) from 2011. Instructions to sign up or download data are contained within the links. 

##Tips for playing with this code
To get the environment for all the dependencies of this project working, use the `requirements.txt` file. This is done by:
`pip install -r requirements.txt`

## Detailed research
This was done as a project for Andrew Ng's Machine Learning class at Stanford by Timon Ruban, Vincent Sitzmann and Rishabh Bhargava, and if you are interested in learning more about the work, check out the following links:
* Report on using [Random Forests](http://cs229.stanford.edu/proj2015/173_report.pdf) for this problem.
* Nice looking [Poster](http://cs229.stanford.edu/proj2015/173_poster.pdf)

Feel free to contact us at rish93@stanford.edu, sitzmann@stanford.edu or timon@stanford.edu. 
