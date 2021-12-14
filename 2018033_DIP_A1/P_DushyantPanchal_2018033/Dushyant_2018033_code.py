import cv2
import numpy as np
import sys


RUN = ""
if len(sys.argv)>1:
    RUN = sys.argv[1]


## Q3 Solution

def BilinearInterpolation(img, factor, epsilon=1e-4):
    """
    Inputs:
        - img : input image, must be a 2d matrix with grayscale values (0-255)
        - factor : interpolation factor by which we want to scale img
        - epsilon (default 1e-4): value of lambda in inv(A + lambda*I) to ensure we do not try to invert a singular matrix.
    
    Outputs: interpolated version of 'img' by 'factor' amount.
    """
    size = img.shape
    if len(size)!=2:
        return None

    size1 = ( int(factor[0]*size[0]), int(factor[1]*size[1]) )
    img = np.pad(img, (1,1))
    img1 = np.zeros(size1)

    for i in range(size1[0]):
        for j in range(size1[1]):
            
            x = i/factor[0]
            y = j/factor[1]
            
            x1, y1 = round(x), round(y)
            x2, y2 = round(x+1), round(y)
            x3, y3 = round(x), round(y+1)
            x4, y4 = round(x+1), round(y+1)
            
            X = np.array([ [x1, y1, x1*y1, 1], [x2, y2, x2*y2, 1], [x3, y3, x3*y3, 1], [x4, y4, x4*y4, 1] ])
            V = np.array([ img[x1,y1], img[x2,y2], img[x3,y3], img[x4,y4] ])

            A = np.dot( np.linalg.inv( X + epsilon*np.eye(4) ), V )

            val = A[0]*x + A[1]*y + A[2]*x*y + A[3]

            if val>255:
                val = 255
            if val<0:
                val = 0
            
            img1[i,j] = val

    return img1.astype('uint8')

if RUN == "Q3":
    img = cv2.imread("assign1_big.bmp", cv2.IMREAD_GRAYSCALE)
    img1 = BilinearInterpolation(img, (1.5,1.5))
    cv2.imshow("original", img)
    cv2.imshow("interpolated", img1)
    cv2.imwrite("q3_output.png", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


## Q4 Solution

def rotate(angle):
    """ 
    Returns the transformation matrix for rotation by 'angle' radians.
    """
    return np.array([
        [ np.cos(angle), -np.sin(angle), 0 ],
        [ np.sin(angle), np.cos(angle), 0 ],
        [ 0, 0, 1 ]
    ])

def scale(sx, sy):
    """
    Returns the transformation matrix for scaling by sx and sy on the respective axes.
    """
    return np.array([
        [sx,0,0],
        [0,sy,0],
        [0,0,1]
    ])

def translate(tx, ty):
    """
    Returns the transformation matrix for translation by tx and ty on the respective axes.
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [tx, ty, 1]
    ])

def GeometricTransformation(img, T, output_dim=500, epsilon=1e-4, whole_output=False):
    """
    Inputs:
        - img : input image, must be a 2d matrix with grayscale values (0-255)
        - T : 3x3 transformation matrix to be applied on 'img'
        - output_dim (default=500) : maximum size of the output image will be (output_dim x output_dim) 
            and divided into 4 quadrants, each axis ranging from -output_dim/2 to +output_dim/2.
        - epsilon (default 1e-4): value of lambda in inv(A + lambda*I) to ensure we do not try to invert a singular matrix.
        - whole_output (default=False) : whether to display the whole output image of just the part containing the input image.
    
    Outputs the geometrically transformed and bilinearly interpolated image.
    """
    size = img.shape
    if len(size)!=2:
        return None
    
    img1 = np.zeros((output_dim, output_dim))
    T_inv = np.linalg.inv(T)
    img = np.pad(img, (1,1))

    i_min = output_dim
    j_min = output_dim
    i_max = 0
    j_max = 0

    for i in range(output_dim):
        for j in range(output_dim):
            i_ = i - (output_dim//2)
            j_ = j - (output_dim//2)
            X = np.array([i_, j_, 1])
            V = np.dot(X, T_inv)

            x,y = V[0],V[1]

            if x<0 or x>=size[0]:
                continue
            if y<0 or y>=size[1]:
                continue

            i_min = min(i_min, i)
            j_min = min(j_min, j)
            i_max = max(i_max, i)
            j_max = max(j_max, j)

            x1, y1 = round(x), round(y)
            x2, y2 = round(x+1), round(y)
            x3, y3 = round(x), round(y+1)
            x4, y4 = round(x+1), round(y+1)
            
            X = np.array([ [x1, y1, x1*y1, 1], [x2, y2, x2*y2, 1], [x3, y3, x3*y3, 1], [x4, y4, x4*y4, 1] ])
            V = np.array([ img[x1,y1], img[x2,y2], img[x3,y3], img[x4,y4] ])

            A = np.dot( np.linalg.inv( X + epsilon*np.eye(4) ), V )

            val = A[0]*x + A[1]*y + A[2]*x*y + A[3]

            if val>255:
                val = 255
            if val<0:
                val = 0
            
            img1[i,j] = val

    if whole_output:
        return img1.astype("uint8")
    else:
        return img1[i_min:i_max+1, j_min:j_max+1].astype("uint8")

if RUN == "Q4":
    img = cv2.imread("assign1_small.jpg", cv2.IMREAD_GRAYSCALE)
    T = np.dot(np.dot(rotate(np.pi/4), scale(2,2)), translate(30,30))
    print(T)
    img1 = GeometricTransformation( img, T , whole_output=True )
    cv2.imshow("original", img)
    cv2.imshow("transformed", img1)
    cv2.imwrite("q4_output.png", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


## Q5 Solution

if RUN == "Q5a":
    COLORS = [ list(np.random.random(size=3) * 256) for i in range(100) ]

    U = []
    X = []

    def click_event_u(event, x, y, flags, params):
        """
        Event handler for image u to record pixel values for clicked points.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            U.append([y,x,1])
            print(f"U = np.array({U})")
            cv2.circle(img_u, (x,y), 3, COLORS[len(U)], thickness=2)
            cv2.imshow("img_u", img_u)

    def click_event_x(event, x, y, flags, params):
        """
        Event handler for image x to record pixel values for clicked points.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            X.append([y,x,1])
            print(f"X = np.array({X})")
            cv2.circle(img_x, (x,y), 3, COLORS[len(X)], thickness=2)
            cv2.imshow("img_x", img_x)
    
    img_x = cv2.imread("assign1_small.jpg")
    img_u = cv2.imread("transformed.png")
    
    cv2.imshow("img_x", img_x)
    cv2.imshow("img_u", img_u)

    cv2.setMouseCallback('img_x', click_event_x)
    cv2.setMouseCallback('img_u', click_event_u)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def register_image(U, X, img):
    """
    Given U and X from X = UT, find T and apply inverse on the unregistered
    image to obtain and return the registered aligned image.
    """
    T = np.dot( np.linalg.inv( np.dot(U.T, U) ), np.dot(U.T, X) )
    print(np.linalg.inv(T))
    return GeometricTransformation(img, T, whole_output=True)

if RUN == "Q5b":
    img_orig = cv2.imread("assign1_small.jpg")
    img_unreg = cv2.imread("transformed.png", cv2.IMREAD_GRAYSCALE)
    
    X = np.array([[21, 25, 1], [33, 9, 1], [38, 22, 1]])
    U = np.array([[94, 96, 1], [93, 53, 1], [117, 63, 1]])

    img_reg = register_image(U, X, img_unreg)
    
    cv2.imshow("original", img_orig)
    cv2.imshow("unregistered", img_unreg)
    cv2.imshow("registered", img_reg)
    cv2.imwrite("q5_output.png", img_reg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()