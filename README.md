# Classifying Speed Limits

When we first began the project, our goal was to (1) identify speed limit sign from a given image (and video if we could figure out how to do it) and (2) Take that identified speed limit sign and extract the digits from it to get the speed limit value. We wanted to use USA speed limit signs as many solutions out there make use of the German traffic sign dataset. Through the literature review, we came to an understanding that using a shape detection algorithm is best suited for detecting the signs as color based algorithms would be more inaccuracte due to lighting conditions and that using some kind of text reader or classifier to read the numbers is the best way to go.

The main hurdles we faced are:
  1) Many publicly available solutions use the German speed limit sign data which is easier since it only has a number on it and there      are pre exisitng nethods to detect circles in images.
  2) Some solutions were propietary or only briefly presented in a high level manner in scholaraly papers
  3) Lot of initial research and solutions we saw only goes as far as detecting and recognizing what type of sign is seen rather than        just focusing on speed limit signs itself.
  4) When it came to extracting digits, many existing solutions focused on single digit extraction or recognition.
  5) Some possibilities used a laptop camera for recognizing numbers but results differed based on quality of the camera
  6) The biggest issue of all: lack of USA speed limit signs data/ images for proper training.
  
All these hurdles were a result of us trying 20 plus different models and tweaking it in hope of better performance and suited to our needs.
 
In the end, due to time restrictions and scope, under the advise of our professor, we focused on creating a speed limit sign dataset and attempted to create a neural network program that takes in this dataset and classifies the images. So after finding the LISA dataset, we worked on cropping each of the street view images to only have the speed limit signs itself. The main issue with this is that some of the images were blurry due to the amount of cropping done on the image and the speed limit sign itself being in the far distance in the image. Also, there was a lack of image variety as you will see below. On the high end, we had 587 images for one type of speed limit and 3 images for another type of speed limit on the low end.
 

We created a Tensorflow model that takes in labeled images, trains a tensorflow model for a certain number of iterations and then is tested on the test dataset.
Our dataset consists of images for the speed limit values of 25, 30, 35, 40, 45, 50, & 60. We suspect at the time of the creation of the dataset, there wasn't widespread usage of 70 mph speed limit, thus the lack of it in the LISA Traffic Sign Dataset.
Below is the breakdown of number of images we have for each type of speed limit:
    <ul>25  - 587 images</ul>
    <ul>30  - 157 images</ul>
    <ul>35  - 141 images</ul>
    <ul>40  - 87 images</ul>
    <ul>45  - 77 images</ul>
    <ul>50  - 48 images</ul>
    <ul>60  - 3 images</ul>

The required packages for this program are: tensorflow, IPython, PIL, glob, numpy, cv2, & random.

1) We first created a probability distribution for each unique speed limit value.
  ```
  prob_25 = np.array([1.0, 0.5, 0.25, 0.1, 0, 0, 0])
  prob_30 = np.array([0.5, 1.0, 0.5, 0.25, 0.1, 0, 0])
  prob_35 = np.array([.25, 0.5, 1., 0.5, .25, .1, 0])
  prob_40 = np.array([.1, .25, .5, 1., .5, .25, .1])
  prob_45 = np.array([0, .1, .25, .5, 1., .5, .25])
  prob_50 = np.array([0, 0, .1, 0.25, .5, 1., .5])
  prob_60 = np.array([0, 0, 0, .1, 0.25, .5, 1.])
  ```

2) We read in the data by using a for loop for each type of speed limit. For each speed limit value, the images are stored in a 
   respective folder. We pull each folder, get the image from that folder, resize the image, flatten it, and append it to the image        list and the probability array. Below is an example of this for loop for speed limit value 25.
  ```
  for i, filename in enumerate(glob.glob('25/*.png')):
    prob = np.zeros(7)
    prob = prob_25
    im_full = cv2.imread(filename, 0)
    resized_image = cv2.resize(im_full, (64, 60))
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    resized_image = resized_image.flatten()
    image_list.append(resized_image)
    speed_limit_probs.append(prob)
  ```

3) Then all the data was combined into a list, shuffled, and split into training and testing lists
  ```
  combined = list(zip(image_list, speed_limit_probs))
  random.shuffle(combined)

  image_list[:], speed_limit_probs[:] = zip(*combined)

  training_indices = 900
  testing_indices = 1100

  images_training = image_list[:900] #Normalize between 0 and 1
  speedLimit_training = speed_limit_probs[:900]

  images_testing = image_list[900:1100] #Normalize between 0 and 1
  speedLimit_testing = speed_limit_probs[900:1100]
  ```

4) For the tensorflow model, we used a learning rate of 0.0001, 3000 iterations, 4 layers, relu activation function, Adam Optimizer,      and mean squared error loss

5) After training and running the model on the test dataset. Overall, due to uneven distribution of the data, the model tended to          misclassify iamges as speed limit value 25. In summary, the model had high recall but low accuracy and precision.

In the end, compared to what our original ambition was, we did manage to create a dataset that can be used by others and a model that has the possibility of better performance with improved data variety. This project really helepd us appreciate how important it is to not only have a dataset for the problem one wants to solve, but also a good dataset. Hopefully in the future, there will be more USA speed limit sign datasets available as the research in driverless cars and computer vision in automation is continuing to expand and become more accessible to the public in an open source manner.
