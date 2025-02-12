from typing import Union

import SimpleITK as sitk


def overlay(image, label_map, color_map: Union[list, tuple] = None, opacity=0.5):
    image = sitk.GetImageFromArray(image)
    label_map = sitk.GetImageFromArray(label_map)
    if color_map:
        if len(color_map) == 1:
            itk_color_map = list(color_map[0])
        else:
            itk_color_map = list(color_map[-1])
            for i in color_map[:-1]:
                itk_color_map += list(i)
        overlay_image = sitk.LabelOverlay(image, label_map, opacity=opacity, colormap=itk_color_map)
    else:
        overlay_image = sitk.LabelOverlay(image, label_map, opacity=opacity)
    overlay_image = sitk.GetArrayFromImage(overlay_image)
    return overlay_image
