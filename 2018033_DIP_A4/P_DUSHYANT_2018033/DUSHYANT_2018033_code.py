import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) <= 1:
    quit()

RUN = sys.argv[1]


def bgr_to_hsi(img):
    """ Converts input RGB space image to HSI space.
    Input:
        - img : nxmx3 matrix, values in range 0-255
    """
    img = img.astype('float')/255
    out = np.zeros(img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            b, g, r = img[x, y, :3]

            # black color
            if b == g == r == 0:
                continue

            r_g = r - g
            r_b = r - b
            g_b = g - b

            # computing hsi components
            tmp = r_g*r_g + r_b*g_b
            tmp = 0.5*(r_g+r_b) / np.sqrt(tmp + 0.00001)
            tmp = np.clip(tmp, -1, 1)
            theta = np.arccos(tmp)

            h = theta/(2*np.pi)
            if b > g:
                h = 1-h

            s = 1 - 3*min((r, g, b))/(r+g+b)
            i = (r+g+b)/3

            # assigning hsi pixel values
            out[x, y, 0] = h
            out[x, y, 1] = s
            out[x, y, 2] = i

    return out


def hsi_to_bgr(img):
    """ Converts input HSI space image to RGB space.
    Input:
        - img : nxmx3 matrix, values in range 0-255
    """
    img = img.astype('float')/255
    out = np.zeros(img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            h = img[x, y, 0]
            s = img[x, y, 1]
            i = img[x, y, 2]

            # computing the rgb components
            r, g, b = 0, 0, 0
            if 0 <= h < (1/3):
                h = h*2*np.pi
                b = i*(1-s)
                r = i*(1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                g = 3*i - (r+b)
            elif (1/3) <= h < (2/3):
                h -= (1/3)
                h = h*2*np.pi
                r = i*(1-s)
                g = i*(1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                b = 3*i - (r+b)
            elif (2/3) <= h <= 1:
                h -= (2/3)
                h = h*2*np.pi
                g = i*(1-s)
                b = i*(1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                r = 3*i - (r+b)

            # populating the pixel values
            out[x, y, 0] = b
            out[x, y, 1] = g
            out[x, y, 2] = r

    return out


def histogram_equalization(img, channel=2, L=256):
    """
    Input
        - img : image matrix (np.array)
        - channel : channel to use for equalization
        - L (default=256) : integer specifying the number of quantization levels.

    Returns the histogram equalized version of img.
    """
    m, n = img.shape[:2]

    # histogram
    hist = np.zeros(L)
    for i in range(m):
        for j in range(n):
            hist[img[i, j, channel]] += 1

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
    out = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j, channel] = transform[img[i, j, channel]]

    return out


def calc_clsf(shape, lamb=0.25):
    """
    Calculates CLSF filter of given 'shape' assuming
     11x11 box filter degradation and laplacian with
     origin at the top left.
    """
    m, n = shape
    # assumed degradation function
    h = np.pad(np.ones((11, 11))/121, ((0, 2*m-11), (0, 2*n-11)))
    H = np.fft.fft2(h)

    # laplacian filter
    l4 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    l4 = np.pad(l4, ((0, 2*m-3), (0, 2*n-3)))
    L = np.fft.fft2(l4)

    # calculating CLSF
    H_star = H.conjugate()
    H_mag = np.abs(H)
    L_mag = np.abs(L)
    Filter = H_star / (H_mag**2 + lamb*L_mag**2)

    return Filter


if __name__ == '__main__':

    if RUN == "Q1":
        img = cv2.imread("noiseIm.jpg", cv2.IMREAD_GRAYSCALE).astype('float')
        img_ref = cv2.imread(
            "cameraman.jpg", cv2.IMREAD_GRAYSCALE).astype('float')
        m, n = img.shape

        # given noisy signal
        g = np.pad(img, ((0, m), (0, n)))
        G = np.fft.fft2(g)

        # calculating the filter
        Filter = calc_clsf((m, n), lamb=0.25)

        # plotting filter
        Filter_mag = np.abs(Filter)
        Filter_mag_norm = (Filter_mag - np.min(Filter_mag))/np.max(Filter_mag)
        cv2.imshow("CLSF", Filter_mag.astype('uint8'))
        cv2.imshow("CLSF Normalized", Filter_mag_norm)

        # applying the filter
        F_cap = Filter*G
        f_cap = (np.fft.ifft2(F_cap).real)[:m, :n]
        f_cap = np.clip(f_cap, 0, 255)

        # calculating MSE, PSNR
        mse = np.mean((img_ref-f_cap)**2)
        psnr = 10*np.log10(255*255/mse)

        # results
        print("MSE: ", mse)
        print("PSNR: ", psnr)
        cv2.imshow("input image", img.astype('uint8'))
        cv2.imshow("output image", f_cap.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if RUN == "Q3":
        img = cv2.imread("rgbIm.tif")

        hsi_img = bgr_to_hsi(img)
        hsi_img = (hsi_img*255).astype('int')
        cols, hist = np.unique(hsi_img[:, :, 2], return_counts=True)
        plt.plot(cols, hist, "--", label="Before")

        hsi_img_eq = histogram_equalization(hsi_img)
        cols, hist = np.unique(hsi_img_eq[:, :, 2], return_counts=True)
        plt.plot(cols, hist, "--", label="After")

        bgr_img = hsi_to_bgr(hsi_img_eq)

        plt.legend()
        plt.show()

        cv2.imshow("input image", img)
        cv2.imshow("output image", bgr_img)
        print(hsi_img[100, 100])
        print(hsi_img_eq[100, 100])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
