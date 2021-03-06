import numpy as np
import cv2
from skimage.filters import threshold_local
import math
from scipy import ndimage
from os import listdir
from os.path import isfile, join, splitext
import pathlib
from itertools import chain

print("Imports are Done!")


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)  


class ImageFitter:
    """
    A class to Straighten and crop an image

    ...

    Attributes
    ----------
    img : str
        image name of the document/photo.

    Methods
    -------

    Rotation():
        Automatically rotates the image/document to a straight (top-down, face-on) view.

    """    

    def __init__(self, img):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
        img : str
            image name of the document/photo

        Returns
        -------
        None
        """
        
        self.filename = img
        self.img = cv2.imread(img)


    def Crop(self, save_cropped=False):
        print("Croping")
        # read the original image, copy it,
        # rotate it
        image = self.img
        orig = image.copy()
        
        blur = cv2.blur(orig,(15,15))
        imageGrayscale = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        im_bw_245 = cv2.threshold(imageGrayscale, 245, 255, cv2.THRESH_BINARY)[1]

        img_edges_160000_160000_7 = cv2.Canny(im_bw_245, 160000, 160000, apertureSize=7)  
        
        #cv2.imshow("im_bw_245", ResizeWithAspectRatio(im_bw_245,1000))
        #cv2.imshow("img_edges_160000_160000_7", ResizeWithAspectRatio(img_edges_160000_160000_7,1000))
        #cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(img_edges_160000_160000_7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        imageWithContours = image.copy()
        #cv2.drawContours(image=imageWithContours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
        #cv2.drawContours(imageWithContours, contours, -1, (0, 255, 0), 3)
        #cv2.imshow("imageWithContours", ResizeWithAspectRatio(imageWithContours,1000))
        #cv2.waitKey(0)

        #merga all contours into one
        list_of_pts = [] 
        for ctr in contours:
            list_of_pts += [pt[0] for pt in ctr]
        x,y,w,h = cv2.boundingRect(cv2.convexHull(np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)))

        #imageWithContours = image.copy()
        #cv2.rectangle(imageWithContours, (x,y), (x+w, y+h), (0, 255, 0), 10)
        #cv2.imshow("imageWithContours", ResizeWithAspectRatio(imageWithContours,1000))
        #cv2.waitKey(0)

        image = image[y:y+h,x:x+w]

        if save_cropped:
            destFileName = splitext(self.filename)[0] + '_cropped.png'
            # Saving an image itself
            cv2.imwrite(destFileName, image)

        self.img = image
        return image


    def Rotation(self, save_rotated=False):
        """
        Rotate an image/document view for a face-on view (view from the top).

        Optionally, saves and resizes a collage with the original and scanned images.

        Parameters
        ----------
        save_rotated : bool
            flag to save the rotated image
        resize_height : int (optional, default = 500)
            final height to resize an image to (in pixels)

        Returns
        -------
        Rotated image (array)
        """

        print("Rotation")
        # read the original image, copy it,
        # rotate it
        image = self.img
        orig = image.copy()
        
        imageGrayscale = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        im_bw_245 = cv2.threshold(imageGrayscale, 245, 255, cv2.THRESH_BINARY)[1]

        img_edges_160000_160000_7 = cv2.Canny(im_bw_245, 160000, 160000, apertureSize=7)  

        #cv2.imshow("img_edges_160000_160000_7", ResizeWithAspectRatio(img_edges_160000_160000_7,1000))

        #cv2.imshow("imageGrayscale", imageGrayscale)
        #cv2.waitKey(0)
        
        
        lines = cv2.HoughLinesP(img_edges_160000_160000_7, rho=1, theta=np.pi / 180.0, threshold=160, minLineLength=100, maxLineGap=10)
        
        imageWithLines = image.copy()
        maxLen = 0
        maxAngle = 0
        for [[x1, y1, x2, y2]] in lines:
            # Drawing Hough lines
            cv2.line(imageWithLines, (x1, y1), (x2, y2), (128,0,0), 10)
            length = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
            #lengths.append(length)
            if(length > maxLen) :
                maxLen = length
                maxAngle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        #cv2.imshow("img_edges_200_200", ResizeWithAspectRatio(img_edges_200_200,1000))
        #cv2.imshow("img_edges_300_300", ResizeWithAspectRatio(img_edges_200_200,1000))
        #cv2.imshow("imageWithLines", ResizeWithAspectRatio(imageWithLines,1000))
        #cv2.waitKey(0)

        print("MaxLen: " + str(maxLen))
        print("Rotation Angle: " + str(maxAngle))
        maxAngle %= 90
        if maxAngle > 45:
            maxAngle -= 90
        print("Rotation Angle: " + str(maxAngle))
        # actual rotation
        image = ndimage.rotate(image, maxAngle, cval=255)

        if save_rotated:
            destFileName = splitext(self.filename)[0] + '_rotated.png'
            # Saving an image itself
            cv2.imwrite(destFileName, image)

        self.img = image
        return image


if __name__=="__main__":
    myPath = pathlib.Path().resolve()
    filesPath = join(myPath, "sampleSingleImages")
    files = [join(filesPath, f) for f in listdir(filesPath) if isfile(join(filesPath, f))]
    print(files)

    # to speed debug
    #files = files[slice(1)]

    print(files)

    for imgPath in files:
        print(imgPath)
        fitter = ImageFitter(imgPath)
        rotated_im = fitter.Rotation(save_rotated=False)
        cropped_im = fitter.Crop(save_cropped=True)
    
