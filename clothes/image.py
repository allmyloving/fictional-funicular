from scipy.misc import imread
im = imread("sock.jpg", flatten=True)
print(im.shape)