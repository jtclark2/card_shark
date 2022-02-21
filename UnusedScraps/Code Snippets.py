# import cv2
# import numpy as np
#
# class CodeSnippets():
#     """
#     Cheat sheet filled with code snippets that I didn't end up using
#     Mostly worked, but I snipped them to move here, and didn't bother
#     pulling dependencies along. So they illustrate concepts, but
#     generally won't function without some setup.
#     """
#
#     #Show image and wait for key press
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#
#     #Collect statistics on masked image
#     (mean, std_dev,) = cv2.meanStdDev(gray, mask=mask)
#
#     # Masking
#     masked_img = cv2.bitwise_and(gray, gray, mask=mask)
#
#     #Unwrap image into an array
#     data = masked_img.ravel()
#
#     #conditional removal of data from array
#     data = data[data != 0]  # remove zeros
#
#     #1D Array stats
#     mean = np.mean(data)
#     std_dev = np.std(data)
#     total_intensity = cv2.sumElems(data)
#
#     #fresh copy (rather than reference) of image
#     gray.copyTo(masked_gray, mask)
#
#
#
#
#     # Gather histogram statistics
#     # Useful Snippet: https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
#     channel = [0]  # only a single channel on grayscale
#     bin_count = [256]  # this is full-scale (1 per possible uint8 intensity
#     hist_range = [0, 256]  # ranging from 0 up to the max possible value of uint8
#     # input data can be 1D, or 2D
#     hist_full = cv2.calcHist([data],
#                              channel,
#                              None,
#                              bin_count,
#                              hist_range)
#
#     # Plotting multiple images on one window
#     plt.subplot(221), plt.imshow(gray, 'gray')
#     plt.subplot(222), plt.imshow(mask, 'gray')
#     plt.subplot(223), plt.imshow(masked_img, 'gray')
#     plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
#     plt.xlim(hist_range)
#     plt.show(False)  # false - keeps plt from blocking (True waits for user to close)
#
#
#     # Draw Contours
#     mask = np.zeros(image.shape[:2], np.uint8)
#     mask[:, :, :] = 255 # change mask color
#
#     mask[100:300, 100:400] = 255 #Fill the mask area
#     cv2.drawContours(mask, [c], -1, 255, -1)
#
#
#     contourIdx = -1 # Draw all contours
#     color = 1
#     cv2.drawContours(mask, [c], -1, 255, -1)
#
#     inverse_mask = np.zeros(shape)
#     inverse_mask[:] = 255
#     cv2.fillPoly(inverse_mask, pts=contours, color=(0))
#     normalizer = (masked_gray + inverse_mask).min()
#
