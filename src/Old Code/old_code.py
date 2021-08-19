# basic filter
# filter_shape = cv2.MORPH_RECT
# filter_dilation_size = 4
# filter_size = filter_dilation_size * 2
# filter_size = (filter_size, filter_size)
# filter_cv2 = cv2.getStructuringElement(filter_shape, filter_size)

# filter_cv2 = filter_cv2 / filter_cv2.size

# u_filtered = cv2.filter2D(u_subspace, -1, filter_cv2)
# v_filtered = cv2.filter2D(v_subspace, -1, filter_cv2)

# fig = plt.figure(figsize=(24, 18))
# plt.quiver(x_subspace, y_subspace, u_filtered, v_filtered, color="r")
# plt.show()

def get_filtered_image(event_data, t_end,
                       start_x_img, start_y_img, stop_x_img, stop_y_img, 
                       filter_amount, filter_apothem, f0x,
                       temporal_mono_filter, temporal_bi1_filter, temporal_bi2_filter,
                       scale_biphasic1, scale_biphasic2,
                       spatial_even_filters, spatial_odd_filters
                      ):
    
    # Flip filter axes so neuron's response can be modeled.
    # For example, a neuron in (-10, -10) will response as in (10, 10)
    # and vice versa

    even_filters = np.flip(spatial_even_filters, axis=(1,2))
    odd_filters = np.flip(spatial_odd_filters, axis=(1,2))
    
    even_filters = np.moveaxis(even_filters, 0, 2)
    odd_filters = np.moveaxis(odd_filters, 0, 2)
    
    
    pixels_x = stop_x_img - start_x_img
    pixels_y = stop_y_img - start_y_img
    
    # order is reversed for band_height - width for x, y indexing as in a picture
    grid_vox = np.zeros((pixels_y, pixels_x, filter_amount), dtype=np.float64)
    
    stop_index = len(event_data)
    
    filter_size = 2 * filter_apothem + 1
    
    for index in np.arange(0, stop_index):
        t, x, y = event_subset[index]

        # Compute temporal filter

        t_diff = t_end - t

        temporal_monophasic = temporal_mono_filter.get(t_diff)
        temporal_biphasic = scale_biphasic1 * temporal_bi1_filter.get(t_diff)
        temporal_biphasic += scale_biphasic2 * temporal_bi2_filter.get(t_diff)

        x_start, x_stop, x_filter_start, x_filter_stop = \
            get_axis_indices(x, 0, band_width, filter_apothem, filter_size)
        y_start, y_stop, y_filter_start, y_filter_stop = \
            get_axis_indices(y, 0, band_height, filter_apothem, filter_size)

        even_filter_val = temporal_biphasic * even_filters
        odd_filter_val = temporal_monophasic * odd_filters        
        filter_val = even_filter_val + odd_filter_val

        grid_vox[y_start:y_stop, x_start:x_stop] += \
            filter_val[y_filter_start:y_filter_stop, x_filter_start:x_filter_stop]
    
    return grid_vox

