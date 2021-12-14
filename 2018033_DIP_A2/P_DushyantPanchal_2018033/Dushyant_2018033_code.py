import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


def plot_histogram(img, label=None, L=256):
    """
    Input
        - img : grayscale image matrix (np.array)
        - label (default=None) : provide a plot label if needed (string)
        - L (default=256) : integer specifying the number of quantization levels.
    
    Returns None. Plots the histogram corresponding to img.
    Make sure to call plt.plot() separately after calling this function.
    """
    m,n = img.shape
    
    # calc histogram
    hist = np.zeros(L)
    for i in range(m):
        for j in range(n):
            hist[img[i,j]] += 1
    
    # normalize
    hist = hist/(m*n)

    if label:
        plt.plot(list(range(L)), hist, "--", label=label)
    else:
        plt.plot(list(range(L)), hist, "--")

def histogram_equalization(img, L=256):
    """
    Input
        - img : grayscale image matrix (np.array)
        - L (default=256) : integer specifying the number of quantization levels.
    
    Returns the histogram equalized version of img.
    """
    m,n = img.shape
    
    # histogram
    hist = np.zeros(L)
    for i in range(m):
        for j in range(n):
            hist[img[i,j]] += 1
    
    # probability / normalize
    prob = hist/(m*n)

    # cdf
    cdf = np.zeros(L)
    cdf[0] = prob[0]
    for i in range(1, L):
        cdf[i] = cdf[i-1] + prob[i]

    # transform
    transform = (L-1)*cdf

    # generate the output
    out = np.zeros(img.shape)
    for i in range(L):
        out[np.where(img==i)] = transform[i]
    
    return out.astype("uint8")

def gamma_transform(img, gamma, L=256):
    """
    Input
        - img : input grayscale image (np.array)
        - gamma : value of gamma for the gamma transform
        - L (default=256) : number of quantization levels in img
    
    Returns the gamma transformed image.
    """
    
    mapping = np.array(list(range(L)))
    c = (L-1)**(1-gamma)
    mapping = c*(mapping**gamma)
    
    out = np.zeros(img.shape)
    for i in range(L):
        out[np.where(img==i)] = mapping[i]
    
    plt.plot(mapping, label="gamma=0.5 transform")

    return out.astype("uint8")

def histogram_matching(img, target, L=256):
    """
    Input
        - img : input grayscale image (np.array) to be manipulated.
        - target : target grayscale image (np.array) for histogram reference.
        - L (default=256) : integer specifying the number of quantization levels.
    
    Returns the histogram matched (to target) version of img.
    """
    m,n = img.shape
    
    # histogram
    h = np.zeros(L)
    g = np.zeros(L)
    for i in range(m):
        for j in range(n):
            h[img[i,j]] += 1
            g[target[i,j]] += 1
    
    # probability / normalize
    h = h/(m*n)
    g = g/(m*n)
    
    # cdf
    H = np.zeros(L)
    G = np.zeros(L)
    H[0] = h[0]
    G[0] = g[0]
    for i in range(1, L):
        H[i] = H[i-1] + h[i]
        G[i] = G[i-1] + g[i]
    
    # transform
    transform = []
    for i in range(L):
        map_to = i
        min_diff = float("inf")
        for j in range(L):
            new_diff = abs(H[i] - G[j])
            if new_diff < min_diff:
                min_diff = new_diff
                map_to = j
        transform.append(map_to)
    plt.plot(transform, label="calculated transform")

    # generate the output
    out = np.zeros(img.shape)
    for i in range(L):
        out[np.where(img==i)] = transform[i]
    
    return out.astype("uint8")

def convolution(img, filter):
    """
    Input
        - img : input image matrix f(x,y) (np.array)
        - filter : convolution filter matrix w(x,y) (np.array)
    
    Returns the convolution result f(x,y)*w(x,y).
    """
    m,n = img.shape
    p,q = filter.shape

    # padding image
    print("Input", img.shape)
    print(img)
    print()

    img = np.pad(img, ( (p-1,p-1), (q-1,q-1) ) )

    print("Padded-Input", img.shape)
    print(img)
    print()

    # rotating filter
    print("Input-Filter", filter.shape)
    print(filter)
    print()
    
    filter = np.rot90(np.rot90(filter))
    
    print("Rotated-Filter", filter.shape)
    print(filter)
    print()

    # init output image matrix
    out = np.zeros( (m+p-1, n+q-1) )
    out_dim = out.shape
    
    # iterating over all pixels in output matrix
    for i in range(out_dim[0]):
        for j in range(out_dim[1]):

            val = 0

            # iterating over filter
            for a in range(p):
                for b in range(q):
                    val += img[i+a,j+b]*filter[a,b]
            
            # assigning output pixel value
            out[i, j] = val
    
    print("Convolution-Output", out.shape)
    print(out)

    return out


if __name__=="__main__":
    RUN = ""
    if len(sys.argv)>1:
        RUN = sys.argv[1]
    
    if RUN=="Q3":
        img = cv2.imread("input image.jpg", cv2.IMREAD_GRAYSCALE)
        out = histogram_equalization(img)

        plot_histogram(img, "Original")
        plot_histogram(out, "Equalized")
        plt.legend()
        plt.show()

        cv2.imshow("Input", img)
        cv2.imshow("Equalized", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if RUN=="Q4":
        img = cv2.imread("input image.jpg", cv2.IMREAD_GRAYSCALE)
        target = gamma_transform(img, 0.5)
        out = histogram_matching(img, target)
        plt.legend()
        plt.show()

        plot_histogram(img, "Histogram-Input")
        plot_histogram(target, "Histogram-Target")
        plot_histogram(out, "Histogram-Output")
        plt.legend()
        plt.show()
        
        cv2.imshow("Input", img)
        cv2.imshow("Target", target)
        cv2.imshow("Output", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if RUN=="Q5":
        img = np.array([
            [0,0,0],
            [0,1,0],
            [0,0,0]
        ])
        filter = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])
        convolution(img, filter)
