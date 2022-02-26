from DebrisCounterAnalyze import DebrisCounterAnalysis

# NEXT UP

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-29 Wegner Creek") # eliminates spurious extra early detections due to stuff stuck in the chamber, and a small set in middle of range due to busted-up detritus pulse
# result.detect_and_save_particles()
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[[1831,1981],[2982,2988]])

# result.update_qc_centroids()




# IN PROGRESS


# DONE

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-17 First Bridge #3")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-17 First Bridge #4")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-17 First Bridge #5")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-18 First Bridge #8")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-18 First Bridge #7")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-28 Little Chena")
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-15 Third Bridge") # lots of bubble image exclusions
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.05, do_overlays=False, spreadsheet_count=200, exclude_images=[7467, 7015, 8062, 7476, 7789, 6992, 7661, 8426, 8284, 8496, 7305, 7824, 7391, 6935, 6977, 7016])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-30 First Bridge #3") # one exclusion due to fish
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200, exclude_images=[7804])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-30 First Bridge #4") # one exclusion due to fish
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-30 First Bridge #5") # one exclusion due to fish
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.999, do_overlays=False, spreadsheet_count=200)



result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-29 Little Chena")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-29 First Bridge #1")
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-29 First Bridge #2")
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-14 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.05, do_overlays=False, spreadsheet_count=200, exclude_images=[3209, 4327, 4197, 4824, 3252, 3271, 4273, 4126, 4411, 3427, 3252, 4402])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-14 Nordale")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.95, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-12 Third Bridge") # several individual image exclusions from bubble clouds that weren't all easily detected
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[5971, 5972, 6813, 6814, 5765, 5766, 5923])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-09 First Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-12 First Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-09 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-07 First Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)




result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-05 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[[2074,2080]])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-05 First Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-07-19 Third Bridge") # on this one, excluding lots of the images with the most debris because it's coming in big slugs of periphyton breaking up
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[4801, 4409, 5005, 4506, 5006, 3719, 4101, 4508, 3774, 4853, 4814, 4845, 5007, 4816, 3822])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-30 First Bridge #7")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.8, do_overlays=False, spreadsheet_count=200, exclude_images=[3280])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-17 Nordale")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-07-05 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.8, do_overlays=False, spreadsheet_count=200, exclude_images=[[2100,2750], 3085, 3630])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-29 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.8, do_overlays=False, spreadsheet_count=200, exclude_images=[2793])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-30 First Bridge #8")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.8, do_overlays=False, spreadsheet_count=200, exclude_images=[3967, 4986])


result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-18 Little Chena")
# result.update_qc_centroids()
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-03 Third Bridge")
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-19 Dam")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)



result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-30 First Bridge #8")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.8, do_overlays=False, spreadsheet_count=200, exclude_images=[3967, 4986])



result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-26 First Bridge #1")
# result.detect_and_save_particles()
result.filter_and_summarize_particles(max_allowable_bubble_probability=1.0, do_overlays=False, spreadsheet_count=200, exclude_images=[378,421,604,632,1135,1671,1732,1808,1823])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-26 First Bridge #2")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[3301, 2954, 3201, 3119, 3539, 3205, 3125, 2637, 3623, 3518])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-26 First Bridge #3")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[4786, 5218, 4329, 5127, 5710, 5101, 5311, 4199, 4200, 5130, 4490, 4927, 5303, 5256, 5128, 4426, 4927, 5231, 4710])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-26 First Bridge #4")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[7236, 6286, 6340, 7266, 7012, 5749, 6341, 6819, 6820, 7011, 6338, 6342, 5796, 5840, 6434, 6340, 5922, 7267, 5922, 7268, 6342, 5796, 7139, 7017, 6342, 6435, 5783, 6343, 5797, 6820, 5795, 5758])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-26 First Bridge #5")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[9006, 8589, 8542, 8671, 9005, 8024, 8231, 8227, 8246, 8386, 8025, 8483, 9007, 8130, 8231, 7476])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-27 First Bridge #6")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[10356])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-27 First Bridge #7")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[12416, 12452, 12417, 12058, 12516, 12502, 11340, 12512, 12056, 12450, 12452])

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-06-27 First Bridge #8")
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200, exclude_images=[1730, 1328, 1332, 1458, 741, 1077, 1099, 1625, 302, 1729, 1619, 1328, 1332, 1325, 1732, 1623, 1326, 1789, 1331, 1329, 303, 1330, 1732, 1328, 1327, 1097, 1733, 1329, 1447, 1327, 1098, 1625])

# PROBLEM ONES

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-05 Nordale") # random turbidity clouds showing up as large debris
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-06 Nordale") # debris cloud problems
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-08-15 Nordale") # todo still has blank screen blocks showing as debris
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.05, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2020-06-23 Nordale") # todo turbidity cloud shenanigans here
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.05, do_overlays=False, spreadsheet_count=200)

result = DebrisCounterAnalysis("/Volumes/DebrisImgWA/", "2019-05-27 Dam") # so much debris lots of aggregates get picked up as big singles
result.filter_and_summarize_particles(max_allowable_bubble_probability=0.99, do_overlays=False, spreadsheet_count=200)
