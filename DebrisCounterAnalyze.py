import os
import cv2
import shutil
import json
import xlsxwriter
import string
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt

NUM_MEDIAN_IMAGES = 5
MEDIAN_DILATION_KERNEL_SIZE = 5
BINARIZATION_THRESHOLD = 5  # 10 was working pretty well
PIXELS_PER_MM = 24.1
VOLUME_SAMPLED_PER_IMAGE_M3 = 0.002914
IMAGE_DIMENSIONS = (7952, 5304)
MEAN_BUBBLE_COLOR_NORMALIZED = np.array([1.07, 0.97, 0.96])

RESULTS_SHARED_FOLDER = os.path.join(os.path.expanduser("~"), 'Dropbox', 'SFR', 'Projects', '2019 Chena Drift Project', 'Chena Drift Project Data', 'Debris Particle Counts')

# Minimum settings: These are in place to prevent the generation of so much detection data that it fills my hard drive and takes
# impossibly long to process everything, while still gathering way more than enough data to characterize the debris in the stream,
# including in larger, rarer size classes. To best characterize the larger size classes means processing every image, at least for
# large debris, but counting every 0.5 mm piece in 2000 debris-dense images would make the result size insane. Thus, we begin by
# recording all debris in the first image, and then stop counting 0.5-1.5 mm items after reaching a given minimum count and
# record only larger ones for the rest of the images, stopping with each size class after that minimum count is reached. The exception
# is if it's reached too early in very few images, in which case I still enforce counting from a minimum number of images so that
# there's not substantial error due to natural variation in counts / debris from one image to the next. That takes priority, so
# we might keep on counting past MAX_PARTICLES_PER_SIZE_CLASS if we have not yet reached MIN_IMAGES_PER_SIZE_CLASS. The number
# of images from which each size class was counted is tracked for volume and total concentration calculations.

MAX_PARTICLES_PER_SIZE_CLASS = 100000
MIN_IMAGES_PER_SIZE_CLASS = 25
DETECTION_OVERLAY_IMAGE_INCREMENT = 10  # Only generate a full detection overlay jpeg for every Nth image

# Scale results:
# From 2019-05-28 Wegner Creek: 2649.47 pixels = 110 mm ----> 24.09 pixels/mm, 3124.87 pixels = 130 mm ---> 24.04 pixels/mm.
# From 2019-05-23 Nordale: 3168.14 pixels = 130 mm --> 24.37 pixels/mm.
# From 2019-05-23 Third Bridge: 3163.58 pixels = 130 mm --> 24.33 pixels/mm.
# From 2019-05-27 Cripple Creek: 3126.58 pixels = 130 mm --> 24.05 pixels/mm.
# From 2019-05-27 Dam: 3131.35 pixels = 130 mm --> 24.09 pixels/mm.
# From 2019-05-29 Nordale: 3131.03 pixels = 130 mm --> 24.08 pixels/mm.
# As a consensus "close enough" result from all these, I'm going with 24.1 pixels/mm.

# Since the image dimensions are 7952x5304, that's 330.51mm x 220.45 mm.
# Assuming 4 cm space separation (check this!!) that's 0.002914 m3 per image, or about 10.5 m3 photographed per hour.

class DebrisCounterAnalysis:

    def __init__(self, folder_location, site_name, first_frame_number=None, last_frame_number=None, excluded_frames=()):
        # Meanings of optional parameters:
        # Files in the folder should be named as follows '2019_08_15 13_34_03 08534.JPG' with the last number being an index number assigned by the camera/download software.
        # That index number (integer) is what's passed to first_frame_number or last_frame_number to exclude bad frames from the beginning or end of the series.
        # Those indices can also be passed individually in the excluded_frames array to exclude a handful of bad frames.
        # I can put other files in the folder with the images but exclude them from analysis by putting an underscore in front of the filename.
        # A special case of this is a mask blocking out regions to exclude from analysis, which must be named "_exclusion_mask.jpg" and be an image that's all white in the
        # area to include/analyze and all black in the area to exclude.
        self.site_name = site_name
        self.image_path = os.path.join(folder_location, site_name)
        self.results_path = os.path.join(folder_location, site_name, "_results")
        self.full_filename = os.path.join(self.results_path, self.site_name + " Detection Data.json")
        self.partial_filename = os.path.join(self.results_path, self.site_name + " Detection Data Partial.json")
        # Here's how the size class counting limit system works (to prevent taking up too much time/space counting millions of the smallest particles).
        # We keep a running total of particles counted by size class for the entire analysis in self.size_class_particle_totals.
        # We only add a detected particle to the totals (and save its contour) if the size_class_counting_flag for its size class is 1.
        # At the end of analyzing a particular image, we first add self.size_class_counting_flags (all 0 or 1) to self.size_class_images_counted.
        # Then, if we've processed the bare minimum images, i.e. if max(self.size_class_images_counted) >= MIN_IMAGES_PER_SIZE_CLASS,
        # we set self.size_class_counting_flags to 0 at any indices where self.size_class_particle_totals >= MAX_PARTICLES_PER_SIZE_CLASS.
        # This will prevent that size class from being counted for the next image. Later on, I will use self.size_class_images_counted
        # divided by the total images counted to estimate volume sampled for each size class and therefore particle concentrations per m3.
        self.size_class_particle_totals = np.zeros(300)  # each size class count will be recorded with the index for its mm size, so 0.5-1.5 is element [1], and so on
        self.size_class_counting_flags = np.ones(300)  # set to 1 when counting, 0 when done counting a given size class
        self.size_class_images_counted = np.zeros(300)  # total images in which each size class was counted

        self.images_with_rejected_bubbles = {}
        self.excluded_images_too_dark = []
        if not os.path.exists(self.results_path): os.makedirs(self.results_path)
        if not os.path.exists(os.path.join(self.results_path, "Detection Images")): os.makedirs(os.path.join(self.results_path, "Detection Images"))
        # if not os.path.exists(os.path.join(self.results_path, "BG Subtracted Images")): os.makedirs(os.path.join(self.results_path, "BG Subtracted Images"))
        self.files = sorted([os.path.join(self.image_path, f) for f in os.listdir(self.image_path) if os.path.isfile(os.path.join(self.image_path, f)) and f[-4:].lower() == ".jpg" and f[0:2] == "20"])
        if first_frame_number is not None:
            self.files = [file for file in self.files if int(file.split(' ')[-1][:-4]) >= first_frame_number]
        if last_frame_number is not None:
            self.files = [file for file in self.files if int(file.split(' ')[-1][:-4]) <= last_frame_number]
        if len(excluded_frames) > 0:
            self.files = [file for file in self.files if int(file.split(' ')[-1][:-4]) <= last_frame_number]
        # The main exclusion mask is used when initially processing images and excludes whole regions of images from analysis at all.
        # This is the preferred and most efficient way to exclude obviously bad regions.
        exclusion_mask_path = os.path.join(self.image_path, "_exclusion_mask.jpg")
        if os.path.isfile(exclusion_mask_path):
            mask_image = np.asarray(cv2.imread(exclusion_mask_path, cv2.IMREAD_GRAYSCALE))
            _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)  # threshold to round off jpeg artifacts
            self.exclusion_mask = mask_image / np.max(mask_image)  # rescale from max of 255 to max of 1, so I can use it as a multiplier
            self.image_proportion_processed = self.exclusion_mask.mean()  # because it's all 0 or 1, the mean of the mask is the proportion of the image at 1, i.e. the proportion not masked out
        else:
            self.exclusion_mask = None
            self.image_proportion_processed = 1
        post_exclusion_mask_path = os.path.join(self.image_path, "_exclusion_mask_post.jpg")
        # The post exclusion mask is applied after processing the image, to eliminate regions with clearly anomalous detections due to
        # problems that were overlooked when creating the initial mask (a blade of grass that was waving around unnoticed, etc). For this one,
        # the centroid of each detected contour is compared with the value of the mask at that position, which is much less efficient
        # than the exclusion mask used during initial processing, but it's a decent band-aid for fixing image sets after processing.
        if os.path.isfile(post_exclusion_mask_path):
            post_mask_image = np.asarray(cv2.imread(post_exclusion_mask_path, cv2.IMREAD_GRAYSCALE))
            _, post_mask_image = cv2.threshold(post_mask_image, 127, 255, cv2.THRESH_BINARY)  # threshold to round off jpeg artifacts
            self.post_exclusion_mask = post_mask_image / np.max(post_mask_image)  # rescale from max of 255 to max of 1, so I can use it as a multiplier
            if self.exclusion_mask is not None:
                self.image_proportion_processed = (self.exclusion_mask * self.post_exclusion_mask).mean()  # because it's all 0 or 1, the mean of the mask is the proportion of the image at 1, i.e. the proportion not masked out
            else:
                self.image_proportion_processed = self.post_exclusion_mask.mean()
        else:
            self.post_exclusion_mask = None


    def get_test_image(self, i):
        # Will return a single background-subtracted gray image for playing around with other algorithms
        initial_median_files = self.files[:NUM_MEDIAN_IMAGES]  # initialize median files
        self.median_image_arrays = [np.asarray(cv2.imread(file, cv2.IMREAD_GRAYSCALE)) for file in initial_median_files]
        self.compute_current_median()
        self.all_contours = []
        grayscale_image = cv2.imread(self.files[i], cv2.IMREAD_GRAYSCALE)
        background_subtracted_image = self.subtract_median_from(grayscale_image)
        return background_subtracted_image

    def detect_and_save_particles(self, max_images=None, exclude_images=None):
        self.process_images(max_images, exclude_images)
        if len(self.excluded_images_too_dark) > 0:
            excluded_image_list_path = os.path.join(self.results_path, self.site_name + " Excluded Images (Too Dark No Flash).txt")
            with open(excluded_image_list_path, "w") as textfile:
                for element in self.excluded_images_too_dark:
                    textfile.write(element + "\n")
        self.save_raw_contours()

    def update_qc_centroids(self, max_allowable_bubble_probability=0.99):
        """ Utility to redo the centroid map for building post-processing exclusion filters in image sets that didn't start with one. """
        self.load_raw_contours(partial=False)
        if self.post_exclusion_mask is not None:
            def is_not_masked_out(contour):
                return self.post_exclusion_mask[contour['center'][1], contour['center'][0]] > 0.5
            self.all_contours = [{
                'image_file': contours_in_image['image_file'],
                'contours': [contour for contour in contours_in_image['contours'] if is_not_masked_out(contour)]
            } for contours_in_image in self.all_contours]
        self.all_non_bubble_contours = [{
            'image_file': contours_in_image['image_file'],
            'contours': [contour for contour in contours_in_image['contours'] if contour['bubble_probability'] <= max_allowable_bubble_probability]
        } for contours_in_image in self.all_contours]
        self.excluded_bubble_contours = [{
            'image_file': contours_in_image['image_file'],
            'contours': [contour for contour in contours_in_image['contours'] if contour['bubble_probability'] > max_allowable_bubble_probability]
        } for contours_in_image in self.all_contours]
        self.plot_particle_centroids(for_masking=True)

    def filter_and_summarize_particles(self, max_allowable_bubble_probability=0.5, spreadsheet_count=200, do_overlays=True, exclude_images=None):
        self.load_raw_contours(partial=False)
        if exclude_images is not None:
            self.exclude_images(exclude_images)
        if self.post_exclusion_mask is not None:
            def is_not_masked_out(contour):
                return self.post_exclusion_mask[contour['center'][1], contour['center'][0]] > 0.5
            self.all_contours = [{
                'image_file': contours_in_image['image_file'],
                'contours': [contour for contour in contours_in_image['contours'] if is_not_masked_out(contour)]
            } for contours_in_image in self.all_contours]
        self.all_non_bubble_contours = [{
            'image_file': contours_in_image['image_file'],
            'contours': [contour for contour in contours_in_image['contours'] if contour['bubble_probability'] <= max_allowable_bubble_probability]
        } for contours_in_image in self.all_contours]
        self.excluded_bubble_contours = [{
            'image_file': contours_in_image['image_file'],
            'contours': [contour for contour in contours_in_image['contours'] if contour['bubble_probability'] > max_allowable_bubble_probability]
        } for contours_in_image in self.all_contours]
        self.export_particle_counts_by_size_class()
        self.plot_detection_trends()
        self.export_extremes_summaries()
        self.plot_particle_centroids('Particle Detection Locations 0.5+ mm')
        self.plot_particle_centroids('Particle Detection Locations 1.5+ mm', 1.5)
        self.plot_particle_centroids('Particle Detection Locations 2.5+ mm', 2.5)
        self.plot_particle_centroids(for_masking=True)
        self.export_particle_spreadsheet(self.all_non_bubble_contours, 'Actual Debris', 'Largest', spreadsheet_count)
        self.export_particle_spreadsheet(self.excluded_bubble_contours, 'Excluded Bubbles', 'Largest', spreadsheet_count)
        self.export_particle_spreadsheet(self.all_non_bubble_contours, 'Actual Debris', 'Random', spreadsheet_count)
        self.export_particle_spreadsheet(self.excluded_bubble_contours, 'Excluded Bubbles', 'Random', spreadsheet_count)
        if do_overlays: self.make_overlays()

    ##################################################################################################################################
    # PRIMARY FUNCTIONS TO IMPORT IMAGES AND COUNT PARTICLES
    # This creates the all_contours array of dictionaries, in which each element contains a filename and the contours of
    # the particles detected therein.
    ##################################################################################################################################

    def compute_current_median(self):
        median = np.median(self.median_image_arrays, axis=0)
        dilation_kernel = np.ones((MEDIAN_DILATION_KERNEL_SIZE, MEDIAN_DILATION_KERNEL_SIZE), np.uint8)
        self.current_median = cv2.dilate(median, dilation_kernel, iterations=1)

    def subtract_median_from(self, image):
        return cv2.subtract(image, self.current_median, dtype=cv2.CV_8UC1)

    def get_contours(self, morphologically_closed_image, background_subtracted_image, color_image, sobel_image):
        # For speed, I'm using a single mask for the whole image and reading intensity max / variance for every pixel within any contour (i.e. the mask)
        # within the bounding rect of the contour of interest. When contours overlap, this will affect results a bit, but it should be good overall,
        # and it avoids a slowdown of 10X or more when creating hundreds of masks for each image, one for each individual contour.

        # Note that preserving the hierarchy data prevents allows preventing occasional false positives from nested contours
        _, contours, hierarchy = cv2.findContours(morphologically_closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        keeper_contours = []
        all_contours_mask = np.zeros_like(background_subtracted_image)
        cv2.drawContours(all_contours_mask, contours, -1, (255, 255, 255), -1)
        blue_channel_image, green_channel_image, red_channel_image = cv2.split(color_image)
        for contour, h in zip(contours, hierarchy[0]):  # note that elements of hierarchy are nested one level deep to match the same # o elements in contours
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            largest_dimension = max(rect[1])
            largest_dimension_mm = largest_dimension / PIXELS_PER_MM
            largest_dimension_mm_rounded = round(largest_dimension_mm)
            if largest_dimension_mm > 0.5 and (self.size_class_counting_flags[largest_dimension_mm_rounded] == 1) and h[3] == -1:
                area = cv2.contourArea(contour)  # checking area > 0 below prevents rare divisions by 0 on particles with non-existent moments
                if area > 0:
                    self.size_class_particle_totals[largest_dimension_mm_rounded] += 1
                    bounding_rect = cv2.boundingRect(contour)
                    values_in_bounding_rect = background_subtracted_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    b_values_in_bounding_rect = blue_channel_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    g_values_in_bounding_rect = green_channel_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    r_values_in_bounding_rect = red_channel_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    mask_values_in_bounding_rect = all_contours_mask[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    pixel_values_in_contour = values_in_bounding_rect[mask_values_in_bounding_rect == 255]
                    b_pixel_values_in_contour = b_values_in_bounding_rect[mask_values_in_bounding_rect == 255]
                    g_pixel_values_in_contour = g_values_in_bounding_rect[mask_values_in_bounding_rect == 255]
                    r_pixel_values_in_contour = r_values_in_bounding_rect[mask_values_in_bounding_rect == 255]
                    mean_color_bgr = list(np.mean(np.array([b_pixel_values_in_contour, g_pixel_values_in_contour, r_pixel_values_in_contour]), axis=1))
                    mean_color_normalized = mean_color_bgr / np.mean(mean_color_bgr)
                    bubble_color_distance = np.linalg.norm(MEAN_BUBBLE_COLOR_NORMALIZED - mean_color_normalized)
                    brightness_max, brightness_skew, brightness_stdev = np.max(np.abs(pixel_values_in_contour)), stats.skew(np.abs(pixel_values_in_contour)), np.std(pixel_values_in_contour)
                    sobel_values_in_bounding_rect = sobel_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
                    sobel_values_in_contour = sobel_values_in_bounding_rect[mask_values_in_bounding_rect == 255]
                    sobel_max, sobel_skew, sobel_stdev = np.max(np.abs(sobel_values_in_contour)), stats.skew(np.abs(sobel_values_in_contour)), np.std(sobel_values_in_contour)
                    # bubble_probability_OLD = 1 / (1 + np.exp(13.55 - 0.1865 * brightness_max + 0.2395 * brightness_stdev - 4.333 * brightness_skew + 18.87 * bubble_color_distance + 1.018 * largest_dimension_mm))

                    bubble_probability = 1 / (1 + np.exp(16.80 - 0.1500 * brightness_max - 6.581 * brightness_skew + 0.02191 * brightness_max * brightness_skew + 1.954 * brightness_skew ** 2 +
                                   6.134 * bubble_color_distance + 0.4341 * brightness_max * bubble_color_distance - 11.35 * brightness_skew * bubble_color_distance - 0.04749 * sobel_max +
                                   0.0003640 * brightness_max * sobel_max - 0.03489 * brightness_skew * sobel_max))

                    moments = cv2.moments(contour)
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    keeper_contours.append({'contour': contour.tolist(),  # convert from numpy array to python list for serialization
                                            'length_pixels': largest_dimension,
                                            'length_mm': largest_dimension_mm,
                                            'area': area,
                                            'perimeter': cv2.arcLength(contour, True),
                                            'center': (center_x, center_y),
                                            'box': box.tolist(),
                                            'b_max': brightness_max,
                                            'b_skew': brightness_skew,
                                            'b_std': brightness_stdev,
                                            's_max': sobel_max,
                                            's_skew': sobel_skew,
                                            's_std': sobel_stdev,
                                            'bubble_color_distance': bubble_color_distance,
                                            'bubble_probability': bubble_probability
                                            })
        return keeper_contours

    def compute_intensity_multipliers(self):
        # This function computes an array of multipliers (in the dimensions of an image, so they can be multiplied by an image) which adjust for the variable intensity resulting
        # from uneven lighting. When applied to the median image, these multipliers just make the lighting relatively even and flat throughout. When applied to the images we're
        # measuring, it should help reduce the disparity in detections between locations near and far from the lights.
        blurred_median = cv2.medianBlur(self.current_median.astype('uint8'), 75)  # first, blur the median image (median of multiple frames) using a median filter (spatial median within one image) to eliminate particles
        mean_of_blurred_median = np.mean(blurred_median)  # get the average intensity of the median-blurred median image
        self.intensity_multipliers = mean_of_blurred_median / blurred_median  # divide that average intensity by the blurred median image values to get an array of multipliers

    def process_images(self, max_images=None, exclude_images=None):
        num_images = len(self.files) if max_images is None else max_images
        self.total_volume_sampled = VOLUME_SAMPLED_PER_IMAGE_M3 * num_images * self.image_proportion_processed
        all_filenames = [os.path.basename(file_path) for file_path in self.files]
        if exclude_images is not None:
            images_to_exclude = [item for item in exclude_images if isinstance(item, int)]
            image_ranges = [item for item in exclude_images if isinstance(item, list) and len(item) == 2]
            for range_start, range_end in image_ranges:
                for i in np.arange(range_start, range_end + 1):
                    images_to_exclude.append(i)
            all_filenames = [filename for filename in all_filenames if int(filename.split(" ")[-1][:5]) not in images_to_exclude]
        if not os.path.exists(self.partial_filename):
            self.all_contours = []
            filenames_to_process = all_filenames
        else:
            self.load_raw_contours(partial=True)
            processed_filenames = [file_data['image_file'] for file_data in self.all_contours]
            filenames_to_process = [filename for filename in all_filenames if filename not in processed_filenames]
        print("Creating initial background image.")
        initial_median_files = filenames_to_process[:NUM_MEDIAN_IMAGES]  # initialize median files
        self.median_image_arrays = [np.asarray(cv2.imread(self.files[all_filenames.index(filename)], cv2.IMREAD_GRAYSCALE)) for filename in initial_median_files]
        self.compute_current_median()
        self.compute_intensity_multipliers()  # might need to also add this in the loop every 200 images or so if lighting is varying
        for image_filename in filenames_to_process:
            i = all_filenames.index(image_filename)
            print(self.site_name, ": Processing image", i + 1, "of", num_images)
            if (i > NUM_MEDIAN_IMAGES / 2) and (i < num_images - NUM_MEDIAN_IMAGES / 2):
                del self.median_image_arrays[0]
                next_median_file = self.files[i + int(np.floor(NUM_MEDIAN_IMAGES / 2))]
                self.median_image_arrays.append(np.asarray(cv2.imread(next_median_file, cv2.IMREAD_GRAYSCALE)))
                self.compute_current_median()
            color_image = cv2.imread(self.files[i], cv2.IMREAD_COLOR)
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            if grayscale_image.mean() < 0.5 * self.current_median.mean():
                self.excluded_images_too_dark.append(image_filename)
                continue  # skip this image if it is so dark it suggests the flash timing was off (will be excluded from processed image counts)
            background_subtracted_image = self.subtract_median_from(grayscale_image)
            intensity_adjusted_image = background_subtracted_image * self.intensity_multipliers
            if self.exclusion_mask is not None:
                intensity_adjusted_image = intensity_adjusted_image * self.exclusion_mask
            sobel_image = cv2.Sobel(background_subtracted_image, cv2.CV_64F, 1, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
            # bg_subtracted_image_output_path = os.path.join(self.results_path, "BG Subtracted Images", os.path.basename(self.files[i])[:-4] + " BG Subtracted.jpg")
            # sobel_image_output_path = os.path.join(self.results_path, "Sobel Images", os.path.basename(self.files[i])[:-4] + " Sobel.jpg")
            # cv2.imwrite(bg_subtracted_image_output_path, intensity_adjusted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            # cv2.imwrite(sobel_image_output_path, sobel_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            _, thresholded_image = cv2.threshold(intensity_adjusted_image, BINARIZATION_THRESHOLD, 255, cv2.THRESH_BINARY)
            closing_kernel = np.ones((3, 3), np.uint8)
            morphologically_closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, closing_kernel).astype('uint8')
            contours = self.get_contours(morphologically_closed_image, intensity_adjusted_image, color_image, sobel_image)
            self.all_contours.append({'image_file': image_filename, 'contours': contours})
            self.size_class_images_counted += self.size_class_counting_flags
            if i >= MIN_IMAGES_PER_SIZE_CLASS:
                size_class_indices_exceeding_max_particles = np.where(self.size_class_particle_totals >= MAX_PARTICLES_PER_SIZE_CLASS)
                self.size_class_counting_flags[size_class_indices_exceeding_max_particles] = 0
            if i % 10 == 0:
                self.save_raw_contours(partial=True)
            if i % 100 == 0:
                self.compute_intensity_multipliers()

    ##################################################################################################################################
    # SAVE RAW RESULTS TO FILE
    ##################################################################################################################################

    def save_raw_contours(self, partial=False):
        print("Saving raw contours to file.")
        saved_data = {
            'all_contours': self.all_contours,
            'total_volume_sampled': self.total_volume_sampled,
            'size_class_images_counted': list(self.size_class_images_counted)
        }
        filename_to_write = self.partial_filename if partial else self.full_filename
        with open(filename_to_write, "w") as output_file:
            json.dump(saved_data, output_file)
        if not partial:
            if os.path.exists(self.partial_filename):
                os.remove(self.partial_filename)

    def load_raw_contours(self, partial):
        print("Loading raw contours from file.")
        saved_contours_path = self.partial_filename if partial else self.full_filename
        if not os.path.exists(saved_contours_path):
            raise ValueError("Tried to load detected contours but file '{0}' not found.".format(saved_contours_path))
        loaded_data = json.loads(open(saved_contours_path, "r").read())
        self.all_contours = loaded_data['all_contours']
        self.total_volume_sampled = loaded_data['total_volume_sampled']
        self.size_class_images_counted = np.array(loaded_data['size_class_images_counted'])

    def redo_bubble_probabilities(self):
        # Only to be used when I've changed the bubble probability formula and want to redo it from the result numbers without redoing every image
        self.load_raw_contours(partial=False)
        print("Recalculating contour bubble probabilities.")
        # test.all_contours[0]['contours'][0]['b_max']
        for image_contour_data in self.all_contours:
            for contour in image_contour_data['contours']:
                brightness_max = contour['b_max']
                brightness_skew = contour['b_skew']
                sobel_max = contour['s_max']
                bubble_color_distance = contour['bubble_color_distance']
                bubble_probability = 1 / (1 + np.exp(16.80 - 0.1500 * brightness_max - 6.581 * brightness_skew + 0.02191 * brightness_max * brightness_skew + 1.954 * brightness_skew ** 2 +
                                                     6.134 * bubble_color_distance + 0.4341 * brightness_max * bubble_color_distance - 11.35 * brightness_skew * bubble_color_distance - 0.04749 * sobel_max +
                                                     0.0003640 * brightness_max * sobel_max - 0.03489 * brightness_skew * sobel_max))
                contour['bubble_probability'] = bubble_probability
        self.save_raw_contours()

    ##################################################################################################################################
    # OVERLAY DETECTIONS ONTO IMAGES
    ##################################################################################################################################

    def make_single_image_overlay(self, image_contour_data):
        color_image = cv2.imread(os.path.join(self.image_path, image_contour_data['image_file']), cv2.IMREAD_COLOR)
        contours_that_arent_bubbles = np.array([contour['box'] for contour in image_contour_data['contours'] if contour['bubble_probability'] <= 0.01])
        contours_maybe_bubbles_1 = np.array([contour['box'] for contour in image_contour_data['contours'] if 0.01 < contour['bubble_probability'] < 0.1])
        contours_maybe_bubbles_2 = np.array([contour['box'] for contour in image_contour_data['contours'] if 0.1 <= contour['bubble_probability'] < 0.9])
        contours_maybe_bubbles_3 = np.array([contour['box'] for contour in image_contour_data['contours'] if 0.9 <= contour['bubble_probability'] < 0.99])
        contours_maybe_bubbles_4 = np.array([contour['box'] for contour in image_contour_data['contours'] if contour['bubble_probability'] >= 0.99])
        # print("Contour counts are {0} non-bubbles, {1} ranking 1, {2} ranking 2, {3} ranking 3, {4} ranking 4. ".format(len(contours_that_arent_bubbles), len(contours_maybe_bubbles_1), len(contours_maybe_bubbles_2), len(contours_maybe_bubbles_3), len(contours_maybe_bubbles_4)))
        cv2.drawContours(color_image, contours_that_arent_bubbles, -1, (8, 254, 101), 1)  # bright lime green color -- not colors are BGR and not RGB!
        cv2.drawContours(color_image, contours_maybe_bubbles_1, -1, (274, 255, 196), 1)   # eggshell blue, maybe but unlikely to be bubbles
        cv2.drawContours(color_image, contours_maybe_bubbles_2, -1, (82, 255, 253), 2)    # lemon, likely to be bubbles in otherwise good images
        cv2.drawContours(color_image, contours_maybe_bubbles_3, -1, (0, 91, 255), 2)   # bright orange, very likely to be bubbles
        cv2.drawContours(color_image, contours_maybe_bubbles_4, -1, (2, 0, 254), 2)   # fire engine red, extremely likely to be bubbles
        image_output_path = os.path.join(self.results_path, "Detection Images", os.path.basename(image_contour_data['image_file'])[:-4] + f" Detections ({len(contours_that_arent_bubbles)+len(contours_maybe_bubbles_2)+len(contours_maybe_bubbles_3)}).jpg")
        cv2.imwrite(image_output_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    def make_overlays(self):
        # Creates overlays of the detected particles in all images at intervals (specified by the constant at the top of the file) as well
        # as those with the most and fewest detections, and those detecting the largest contours, so they can be spot checked for anomalies.
        image_contour_counts = [len(image_data['contours']) for image_data in self.all_contours]
        image_largest_contour_areas = [max([contour['area'] for contour in image_data['contours']], default=0) for image_data in self.all_contours]
        percentile_top_contoursize = np.percentile(image_largest_contour_areas, 99)
        for i, image_data in enumerate(self.all_contours):
            min_index = min(i-100, 0)  # will create overlays for all images with contour counts in the top 5 % of adjacent images
            max_index = max(i+100, len(self.all_contours))
            percentile_bottom = np.percentile(image_contour_counts[min_index:max_index], 1)
            percentile_top = np.percentile(image_contour_counts[min_index:max_index], 99)
            num_contours_in_image = len(image_data['contours'])
            largest_contour_area_in_image = max([contour['area'] for contour in image_data['contours']], default=0)
            if i % DETECTION_OVERLAY_IMAGE_INCREMENT == 0 or num_contours_in_image <= percentile_bottom or num_contours_in_image >= percentile_top or largest_contour_area_in_image >= percentile_top_contoursize:
                print("Making detection overlay image for", image_data['image_file'], "(image", i+1,"of", len(self.all_contours), ").")
                self.make_single_image_overlay(image_data)

    ##################################################################################################################################
    # FUNCTIONS TO EXPORT A SPREADSHEET OF SELECT DETECTED PARTICLES WITH IMAGES AND STATS
    ##################################################################################################################################

    def random_temp_image_path(self):
        letters = string.ascii_lowercase
        filename = ''.join(random.choice(letters) for _ in range(12)) + ".jpg"
        return os.path.join(self.results_path, "Temp", filename)

    def image_of_single_detection(self, image, contour):
        # Based somewhat on this tutorial on how to pull out images of detected particles and rotate appropriately
        # https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        rect = cv2.minAreaRect(np.array(contour))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        # coordinates of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))
        # rotate the image 90 degrees for display in spreadsheet if height > width
        if height > width:
            warped = cv2.transpose(warped)
            warped = cv2.flip(warped, flipCode=1)
        image_path = self.random_temp_image_path()
        cv2.imwrite(image_path, warped)
        height, width, channels = warped.shape
        return image_path, height, width

    def exclude_images(self, which_images):
        """ The input which_contours is a list, with elements being either ints referencing individual images to exclude or lists containing the first
            and last element in a range of images to exclude. """
        images_to_exclude = [item for item in which_images if isinstance(item, int)]
        image_ranges = [item for item in which_images if isinstance(item, list) and len(item) == 2]
        for range_start, range_end in image_ranges:
            for i in np.arange(range_start, range_end+1):
                images_to_exclude.append(i)
        self.all_contours = [contour_image for contour_image in self.all_contours if int(contour_image['image_file'].split(" ")[-1][:5]) not in images_to_exclude]

    def export_contours(self, contours, sheet_title):
        workbook = xlsxwriter.Workbook(os.path.join(self.results_path, '{0} {1}.xlsx'.format(self.site_name, sheet_title)))
        worksheet = workbook.add_worksheet(sheet_title)
        worksheet.write('A1', 'Length (mm)')
        worksheet.write('B1', 'Length (pixels)')
        worksheet.write('C1', 'Area (sq pixels)')
        worksheet.write('D1', 'Perimeter (pixels)')
        worksheet.write('E1', 'Brightness Max')
        worksheet.write('F1', 'Brightness Skew')
        worksheet.write('G1', 'Brightness S.D.')
        worksheet.write('H1', 'Sobel Max')
        worksheet.write('I1', 'Sobel Skew')
        worksheet.write('J1', 'Sobel S.D.')
        worksheet.write('K1', 'Bubble color distance')
        worksheet.write('L1', 'Bubble probability')
        worksheet.write('M1', 'Image (pixels)')
        worksheet.write('N1', 'Filename')
        temp_image_paths = []
        if not os.path.exists(os.path.join(self.results_path, "Temp")): os.makedirs(os.path.join(self.results_path, "Temp"))
        for i, contour in enumerate(contours):
            print("Writing to spreadsheet", sheet_title, "particle", i+1, "of", len(contours), ".")
            row = str(i + 2)
            color_image = cv2.imread(os.path.join(self.image_path, contour['image_file']), cv2.IMREAD_COLOR)
            particle_image_path, height, width = self.image_of_single_detection(color_image, np.array(contour['contour']))
            if i == 0:  # widest particle first
                worksheet.set_column('M:M', width / 7.05)  # Set width of column E
            if height*0.85 > 24:
                worksheet.set_row(i+1, height*0.85)
            temp_image_paths.append(particle_image_path)
            worksheet.write('A' + row, contour['length_mm'])
            worksheet.write('B' + row, contour['length_pixels'])
            worksheet.write('C' + row, contour['area'])
            worksheet.write('D' + row, contour['perimeter'])
            worksheet.write('E' + row, contour['b_max'])
            worksheet.write('F' + row, contour['b_skew'])
            worksheet.write('G' + row, contour['b_std'])
            worksheet.write('H' + row, contour['s_max'])
            worksheet.write('I' + row, contour['s_skew'])
            worksheet.write('J' + row, contour['s_std'])
            worksheet.write('K' + row, contour['bubble_color_distance'])
            worksheet.write('L' + row, contour['bubble_probability'])
            worksheet.insert_image('M' + row, temp_image_paths[-1])
            worksheet.write('N' + row, contour['image_file'])
        worksheet.set_column('A:L', 14)  # Set widths of columns
        worksheet.set_column('N:N', 30)
        workbook.close()
        for path in temp_image_paths: os.remove(path)  # can only remove temp images after closing workbook
        os.rmdir(os.path.join(self.results_path, "Temp"))

    ##################################################################################################################################
    # USE THE ABOVE TO EXPORT A SPREADSHEET OF THE LARGEST PARTICLES DETECTED
    ##################################################################################################################################

    def export_particle_spreadsheet(self, contours, label, which_contours, max_contours):
        single_contours = []
        for image_contour_data in contours:
            for single_contour_data in image_contour_data['contours']:
                single_contours.append(single_contour_data)
                single_contours[-1]['image_file'] = os.path.basename(image_contour_data['image_file'])
        if which_contours == 'Largest':
            single_contours.sort(key=lambda x: -x['length_mm'])
        elif which_contours == 'Random':
            single_contours.sort(key=lambda x: np.random.rand())
        if len(single_contours) <= max_contours:
            final_contours = single_contours
        else:
            final_contours = single_contours[:max_contours]
        if len(final_contours) > 0:
            single_contours.sort(key=lambda x: -x['length_mm'])  # sort by length again for output, after possible randomizing to choose which ones
            self.export_contours(final_contours, which_contours + ' ' + str(max_contours) + ' ' + label)

    ##################################################################################################################################
    # EXPORT PARTICLE COUNTS BY SIZE CLASS
    ##################################################################################################################################

    def export_particle_counts_by_size_class(self):
        size_class_counts = {}
        image_contour_counts = {}
        for image_contour_data in self.all_non_bubble_contours:
            for contour in image_contour_data['contours']:
                length_key = int(round(contour['length_mm']))
                size_class_counts[length_key] = size_class_counts[length_key] + 1 if length_key in size_class_counts.keys() else 1
            image_contour_counts[image_contour_data['image_file']] = len(image_contour_data['contours'])
        workbook_path = os.path.join(self.results_path, '{0} Size Class Counts.xlsx'.format(self.site_name))
        workbook = xlsxwriter.Workbook(workbook_path)
        worksheet = workbook.add_worksheet("Size Class Counts")
        worksheet.write('A1', 'Size Class (mm)')
        worksheet.set_column('A:A', 13)
        worksheet.write('B1', 'Drift Concentration (items/m3)')
        worksheet.set_column('B:B', 26)
        worksheet.write('C1', 'Raw Count')
        worksheet.set_column('C:C', 10)
        worksheet.write('D1', 'Images Counted')
        worksheet.set_column('D:D', 15)
        worksheet.write('E1', 'Volume Sampled (m3)')
        worksheet.set_column('E:E', 15)
        for i, (size_class, count) in enumerate(sorted(list(size_class_counts.items()), key=lambda x: x[0])):
            size_class_volume_sampled = self.total_volume_sampled * self.size_class_images_counted[size_class] / max(self.size_class_images_counted)
            row = str(i + 2)
            worksheet.write('A' + row, size_class)
            worksheet.write('B' + row, count / size_class_volume_sampled)
            worksheet.write('C' + row, count)
            worksheet.write('D' + row, self.size_class_images_counted[size_class])
            worksheet.write('E' + row, size_class_volume_sampled)
        workbook.close()
        shutil.copy2(workbook_path, RESULTS_SHARED_FOLDER)

##################################################################################################################################
# PLOT DETECTION COUNTS CHRONOLOGICALLY AND IN A HISTOGRAM
##################################################################################################################################

    def plot_detection_trends(self):
        detections_per_image = [len(image_contour_data['contours'])for image_contour_data in self.all_non_bubble_contours]
        fig, (ax_time, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
        ax_time.plot(detections_per_image)
        ax_time.set_xlabel("Image # in Sequence")
        ax_time.set_ylabel("Particles Detected per Image")
        ax_hist.hist(detections_per_image, bins=20)
        ax_hist.set_xlabel("Particles Detected per Image")
        ax_hist.set_ylabel("Frequency (# of Images)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'Detections Per Image.pdf'))

##################################################################################################################################
# EXPORT TABLE OF IMAGE FILES WITH THE MOST AND THE FEWEST DETECTIONS
##################################################################################################################################

    def export_extremes_summaries(self):
        detections_per_labeled_image = [(image_contour_data['image_file'], len(image_contour_data['contours'])) for image_contour_data in self.all_non_bubble_contours]
        workbook = xlsxwriter.Workbook(os.path.join(self.results_path, '{0} Images with Most and Least Detections.xlsx'.format(self.site_name)))
        most_sheet = workbook.add_worksheet("Most Detections")
        least_sheet = workbook.add_worksheet("Least Detections")
        most_sheet.write('A1', '# of Detections')
        most_sheet.write('B1', 'Filename')
        most_sheet.set_column('A:A', 15)
        most_sheet.set_column('B:B', 30)
        least_sheet.write('A1', '# of Detections')
        least_sheet.write('B1', 'Filename')
        least_sheet.set_column('A:A', 15)
        least_sheet.set_column('B:B', 30)
        detections_per_labeled_image.sort(key=lambda x: x[1])
        for i, (filename, detection_count) in enumerate(detections_per_labeled_image[:20]):
            row = str(i + 2)
            least_sheet.write('A' + row, detection_count)
            least_sheet.write('B' + row, filename)
        detections_per_labeled_image.sort(key=lambda x: -x[1])
        for i, (filename, detection_count) in enumerate(detections_per_labeled_image[:20]):
            row = str(i + 2)
            most_sheet.write('A' + row, detection_count)
            most_sheet.write('B' + row, filename)
        workbook.close()

##################################################################################################################################
# PLOT PARTICLE CENTROIDS
##################################################################################################################################

    def plot_particle_centroids(self, title=None, min_size_mm=0.5, for_masking=False):
        if not for_masking:  # generate the pdf for easy review, with axes labels etc
            centers = [[contour['center'] for contour in image_contour_data['contours'] if contour['length_mm'] >= min_size_mm] for image_contour_data in self.all_non_bubble_contours]
            centers = [item for sublist in centers for item in sublist]  # flatten the list
            centers_x = [item[0] for item in centers]
            centers_y = list(IMAGE_DIMENSIONS[1] - np.array([item[1] for item in centers]))  # flip up/down from image coordinates for matplotlib
            plt.figure(figsize=(15, 10))
            plt.scatter(centers_x, centers_y, 1.0)
            plt.xlim([0, IMAGE_DIMENSIONS[0]])
            plt.ylim([0, IMAGE_DIMENSIONS[1]])
            if title is not None:
                plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, title + '.pdf'))
            plt.close()
        else:  # generate a jpeg for post-processing mask creation, in the same size as original iamges
            small_centers = [[contour['center'] for contour in image_contour_data['contours'] if contour['length_mm'] <= 1.5] for image_contour_data in self.all_non_bubble_contours]
            small_centers = [item for sublist in small_centers for item in sublist]  # flatten the list
            small_centers_x = [item[0] for item in small_centers]
            small_centers_y = list(IMAGE_DIMENSIONS[1] - np.array([item[1] for item in small_centers]))  # flip up/down from image coordinates for matplotlib
            large_centers = [[contour['center'] for contour in image_contour_data['contours'] if contour['length_mm'] > 1.5] for image_contour_data in self.all_non_bubble_contours]
            large_centers = [item for sublist in large_centers for item in sublist]  # flatten the list
            large_centers_x = [item[0] for item in large_centers]
            large_centers_y = list(IMAGE_DIMENSIONS[1] - np.array([item[1] for item in large_centers]))  # flip up/down from image coordinates for matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(7952/350, 5304/350), dpi=350)
            plt.scatter(small_centers_x, small_centers_y, 1.0)
            plt.scatter(large_centers_x, large_centers_y, 1.0, color='orangered')
            plt.xlim([0, IMAGE_DIMENSIONS[0]])
            plt.ylim([0, IMAGE_DIMENSIONS[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
            plt.savefig(os.path.join(self.results_path, 'Particle Detection Locations for Masking.jpg'), dpi=350)
            plt.close()

# Example usage

# result = DebrisCounterAnalysis("D:/", "2019-08-05 Third Bridge")
# result.detect_and_save_particles()
# result.filter_and_summarize_particles()

# temp = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-10 Third Bridge")
# temp.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False)
