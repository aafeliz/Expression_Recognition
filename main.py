'''
always use the the python 2 in cellar
'''
import dlib as d
import numpy as np
import cv2
import os
import glob
import math as m
import PNN as nn

def knnDistances(point, points):
    distances = []
    for x in points:
        if x is not point:
            distances.append(m.hypot(point[0]-x[0], point[1] - x[1]))
    return distances

# collenction training images from folder
detector0 = d.get_frontal_face_detector()
predictor = d.shape_predictor('shape_predictor_68_face_landmarks.dat')

t_img = []
names = []
masks = []

distances = []
for img in glob.glob("TrainingImages/*.jpg"):
    imgn= cv2.imread(img)
    name = os.path.basename(img).split('.')[0]
    names.append(name)
    t_img.append(imgn)
    faces = detector0(imgn, 1)

    # collecting the 68 feature points on face
    points = []
    for k, j in enumerate(faces):
        shape = predictor(imgn, j)
        points = list(map(lambda p: (p.x, p.y), shape.parts()))

    # create mask from points
    # getting the 68 x 67 distances
    imgDistances = []
    mask = np.zeros((imgn.shape[0],imgn.shape[1]), dtype = np.uint8)
    for x in points:
        #mask[x] = 1
        cv2.circle(mask, (x), 5, (255), thickness=-1)
        y = knnDistances(x, points)
        sum = 0
        for i in y:
            sum = sum + i
        for i in range(len(y)):
            y[i] = y[i]/sum

        imgDistances.append(y)
    distances.append(imgDistances)

    countOnes = 0
    for x in range(720):
        for y in range(1080):
            if mask[x, y] > 0:
                countOnes = countOnes + 1
    print "count is :" + str(countOnes)
    masks.append(np.matrix(mask))

distancesNP = np.array(distances)
distancesNP = np.swapaxes(distancesNP, 0, 1) #4,68,67 to 68,4,67
names = np.array(names)
print(names)
pnn = nn.ParzensNN(distancesNP, distancesNP.shape[0], 0.001, 0.00001, 0.00000001)  # 0.000015
#pnn.test(data)
# TODO: Keep working on a better method to test, and try to get different test data



'''
gray = []
sift = []
for img in t_img:
    gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#sift = cv2.xfeatures2d.SURF_create()
sift = cv2.xfeatures2d.SIFT_create()
kp = []
for i in range(len(masks)):
    imgin = gray[i]
    kp.append(sift.detectAndCompute(gray[i] , masks[i]))

# draw keypoints
kpImages = np.zeros((len(gray), gray[0].shape[0], gray[0].shape[1]))
for j in range(len(masks)):
    #cv2.drawKeypoints(kpImages[j], kp[j], kpImages[j], (100, 255, 0))
    for x in kp[j]:
        cv2.circle(kpImages[j], (int(x.pt[0]), int(x.pt[1])), int(3), (225), thickness=2) #x.size
    cv2.imwrite(names[j]+"kp.jpg", kpImages[j])
    cv2.imwrite(names[j]+"masks.jpg", masks[j])
'''

detector = d.get_frontal_face_detector()
predictor = d.shape_predictor('shape_predictor_68_face_landmarks.dat')
cam = cv2.VideoCapture(0)

#win = d.image_window()
while 1:
    ret, img = cam.read()
    #win.set_image(img)

    faces = detector(img, 1)
    other = img
    for k, j in enumerate(faces):
        shape = predictor(img, j)
        points = list(map(lambda p: (p.x, p.y), shape.parts()))
        #parts = shape.parts.points()
        #part = d.rectangle(shape.rect.left(), shape.rect.top(), shape.rect.right(), shape.rect.bottom())
        #xloc= int((part.left() + part.right())/2)
        #yloc = int((part.top() + part.bottom())/2)
        #cv2.circle(other, (xloc, yloc),10, (0,0,255))
        for x in points:
            cv2.circle(other, (x), 5, (0, 255, 0))
        #xloc = points[0]
        #yloc = points[1]
        #cv2.circle(other, (xloc, yloc), 1, (0, 0, 255))

        #win.clear_overlay
        #win.add_overlay(shape)


    # getting the 68 x 67 distances
    imgDistances = np.zeros(shape=(len(points), (len(points)-1)), dtype=np.float)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    indx = 0
    for x in points:
        # mask[x] = 1
        #cv2.circle(mask, (x), 5, (255), thickness=-1)
        y = knnDistances(x, points)
        sum = 0
        for i in y:
            sum = sum + i
        for i in range(len(y)):
            y[i] = y[i] / sum

        imgDistances[indx] = y
        indx = indx+1
    pnn.test(imgDistances)
    cv2.imshow("results", other)
    cv2.waitKey(30)
cam.release()
cv2.destroyAllWindows()

    #win.add_overlay(faces)

'''
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)


from matplotlib import pyplot as plt
cv2.__version__



'''