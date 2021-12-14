import cv2
import numpy as np
from scipy import signal
import sys

if len(sys.argv)<=1:
    quit()

RUN = sys.argv[1]


# helper function

def display_dft_magnitude(dft, label, normalize=True):
    """
    Displays normalized magnitude spectrum for the provided dft.
    """
    dft_mag = np.abs(dft)
    if not normalize:
        cv2.imshow(label, dft_mag.astype('uint8'))
    else:
        dft_norm = (dft_mag - np.min(dft_mag))/np.max(dft_mag)
        cv2.imshow(label, dft_norm)


# Q1

def butterworth2(D0, shape):
    """
    Returns a butterworth filter of order 2,
    cut off at D0 and of size 2*shape[0] x 2*shape[1].
    """
    m,n = shape
    filter = np.zeros((2*m,2*n))

    for i in range(2*m):
        for j in range(2*n):
            D = ( (i-m)**2 + (j-n)**2 )**0.5
            filter[i,j] = 1/( 1 + (D/D0)**4 )
    
    cv2.imshow(f"2nd Order Butterworth D0={D0}", filter)

    # corresponding spatial filter (only for visualizing)
    sp_filter = np.abs(np.fft.ifft2(filter).real)
    sp_filter = (sp_filter - np.min(sp_filter))/np.max(sp_filter)
    cv2.imshow(f"IDFT of D0={D0}", sp_filter)

    return filter

def conv_fourier_filter(img, filter):
    """
    Performs convolution img*filter in fourier
     space using fourier space filter.

    Returns the result in time domain.
    """
    m,n = img.shape
    img_p = np.pad(img, ((0,m),(0,n)) )
    cv2.imshow("zero padded input img", img_p.astype('uint8'))

    # centering factor (-1)^(m+n)
    centering = np.zeros((2*m,2*n))
    for i in range(2*m):
        for j in range(2*n):
            if (i+j)%2==0:
                centering[i,j] = 1
            else:
                centering[i,j] = -1

    # calculating centered dft
    img_dft = np.fft.fft2(img_p)
    img_dft_cen = np.fft.fft2(img_p*centering)

    display_dft_magnitude(img_dft, "magnitude spectrum of zero-padded input image", normalize=True)
    display_dft_magnitude(img_dft_cen, "centered magnitude spectrum of zero-padded input image", normalize=True)

    # element-wise multiply with the filter
    conv_dft = img_dft_cen*filter

    # inverse dft and take real part
    conv = np.fft.ifft2(conv_dft).real

    # inverse centering
    conv = conv*centering

    # crop from top-left and clip
    return np.clip(conv[:m,:n], 0, 255).astype('uint8')


# Q3

def conv_using_lib(img, filter):
    """
    Returns library based spatial convolution result of img*filter
    """
    lib_conv = signal.convolve2d(img, filter, mode='full').astype('uint8')
    cv2.imshow("lib convolution", lib_conv.astype('uint8'))

    print("Using Library, output size: ", lib_conv.shape)
    return lib_conv

def conv_using_dft(img, filter):
    """
    Returns convolution result of img*filter using DFT
    """
    m,n = img.shape
    p,q = filter.shape

    # zero padding image and filter
    img_p = np.pad(img, ((0,p-1),(0,q-1)) )
    filter_p = np.pad(filter, ((0,m-1),(0,n-1)) )
    cv2.imshow("zero padded input image", img_p.astype('uint8'))

    # calculating dft
    img_dft = np.fft.fft2(img_p)
    filter_dft = np.fft.fft2(filter_p)
    display_dft_magnitude(img_dft, "img dft magnitude", normalize=False)

    # element-wise multiplication
    dft_conv_dft = img_dft*filter_dft
    display_dft_magnitude(dft_conv_dft, "element-wise mul dft magnitude", normalize=False)

    # taking the real part after calculating idft
    dft_conv = np.fft.ifft2(dft_conv_dft).real.astype('uint8')
    cv2.imshow("dft convolution", dft_conv.astype('uint8'))
    
    print("Using DFT, output size:", dft_conv.shape)
    return dft_conv


# Q4

def observation(img, cutoff=0.2):
    """
    Helper function for finding potential points to be filtered.
    """
    m,n = img.shape

    # padding
    img = np.pad(img, ((0,m),(0,n)) )

    # dft magnitude
    dft_mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    
    # min-max normalize
    dft_mag = (dft_mag - np.min(dft_mag))/np.max(dft_mag)

    # display
    cv2.imshow("image normalized dft for observation", dft_mag)

    # candidate noisy points
    points = []
    for i in range(2*m):
        for j in range(2*n):
            if (dft_mag[i,j]>cutoff):
                points.append((i,j))
    print(points)

def noise_filter(noise_points, radius, shape):
    """
    Returns a filter of dimension 'shape' with all ones
     except for points under the 'radius' distance from 
     the 'noise_points'.
    """
    filter = np.ones(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):

            # setting 0 for circle region of size 'radius' around noisy points
            for x,y in noise_points:
                d = ((i-x)**2 + (j-y)**2)**0.5
                if d<radius:
                    filter[i,j] = 0
    
    return filter


if __name__=="__main__":
    
    if RUN=="Q1":
        img = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input image", img)
        cv2.imshow("conv result 10", conv_fourier_filter(img, butterworth2(10, img.shape)))
        cv2.imshow("conv result 30", conv_fourier_filter(img, butterworth2(30, img.shape)))
        cv2.imshow("conv result 60", conv_fourier_filter(img, butterworth2(60, img.shape)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if RUN=="Q3":
        img = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
        filter = np.ones((9,9))/(81)
        cv2.imshow("image", img)
        conv_using_dft(img, filter)
        conv_using_lib(img, filter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if RUN=="Q4":
        img = cv2.imread("noiseIm.jpg", cv2.IMREAD_GRAYSCALE)
        cv2.imshow("image", img)
        m,n = img.shape
        observation(img)
        filter = noise_filter( noise_points=[(192,192), (320,320)], radius=20, shape=(2*m, 2*n) )    
        cv2.imshow("Designed Filter", filter)
        cv2.imshow("Denoised Result", conv_fourier_filter(img, filter))
        cv2.waitKey(0)
        cv2.destroyAllWindows()