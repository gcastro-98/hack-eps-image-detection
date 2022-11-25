import cv2
import numpy as np
from typing import Tuple
# import matplotlib.pyplot as plt
BLACK: tuple = (0, 0, 0)


def retrieve_figures(image_path: str, _visualize: bool = False) -> dict:
    fig_dict = _detect_shapes_and_colors(image_path, _visualize)
    return fig_dict


def __load_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # IF THE BACKGROUND IS TRANSPARENT WE SET IT TO WHITE...
    # make mask of where the transparent bits are
    trans_mask = image[:, :, 3] == 0

    # replace areas of transparency with white and not transparent
    image[trans_mask] = [255, 255, 255, 255]

    # new image without alpha channel...
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def __assert_empty_intersection(lower: dict, upper: dict) -> None:
    for _color in lower.keys():
        # we generate some random points for each color space
        random_color_points = []
        for _ in range(5):
            random_color_points.append(
                [np.random.randint(
                    low=lower[_color][_i], high=upper[_color][_i] + 1)
                    for _i in range(3)])

        # now we check whether it is not in another color space
        for _another_color in [_k for _k in lower.keys() if _k != _color]:
            for _point in random_color_points:
                assertion_message = f"The color {_color} intersects" \
                                    f" with {_another_color}! Check this " \
                                    f"shared element: {_point}"
                assert not all([lower[_another_color][_i] <=
                                _point[_i] <= upper[_another_color][_i]
                                for _i in range(3)]), assertion_message
    return None


def __define_color_bounds(_type) -> Tuple[dict, dict]:
    if _type == 'rgb':
        upper = {
            'red': [255, 100, 100], 'green': [145, 255, 175],
            'blue': [145, 156, 255], 'orange': [255, 130, 100]}
        lower = {
            'red': [50, 0, 0], 'green': [0, 100, 10],
            'blue': [0, 10, 100], 'orange': [175, 50, 0]}
        __assert_empty_intersection(lower, upper)

    elif _type == 'hsv':  # not so properly implemented
        lower = {'red': ([166, 84, 141]), 'green': ([50, 50, 120]),
                 'blue': ([97, 100, 117]), 'orange': ([15, 75, 75])}
        upper = {'red': ([186, 255, 255]), 'green': ([70, 255, 255]),
                 'blue': ([117, 255, 255]), 'orange': ([40, 100, 100])}

    else:
        raise KeyError(f"Unrecognized color code: {_type}")

    return lower, upper


def __create_mask(image, color: str, _type: str = 'rgb'):
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    if _type == 'hsv':
        lower, upper = __define_color_bounds('hsv')
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower[color]), np.array(upper[color]))
        return mask

    lower, upper = __define_color_bounds('rgb')
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, np.array(lower[color]), np.array(upper[color]))
    return mask


def _detect_shapes_and_colors(image_path: str, _visualize: bool = False) -> dict:
    image = __load_image(image_path)
    _fig_dict = {'triangles': 0, 'squares': 0, 'rectangles': 0, 'circles': 0,
                 'red': 0, 'green': 0, 'blue': 0, 'logos': 0}

    mask_list = []  # unused
    hierarchies = []  # unused
    contours = []
    colors_found = []

    for key in ['red', 'green', 'blue', 'orange']:
        kernel = np.ones((2, 2), np.uint8)
        mask = __create_mask(image, key)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_list.append(mask)
        _contour, _hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(_contour) > 0:
            contours.append(_contour[-1])
            colors_found.append(key)
            hierarchies.append(_hierarchy)
            if key != 'orange':  # in this case, 'orange' = 'logo'
                _fig_dict[key] += 1

    # Iterating through each contour to retrieve coordinates of each shape
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        if colors_found[i] == 'orange':
            shape_name = 'Logo'
            _fig_dict['logos'] += 1

        elif len(approx) == 3:
            shape_name = 'Triangle'
            _fig_dict['triangles'] += 1
        elif len(approx) == 4:
            if __is_square(approx):
                shape_name = 'Square'
                _fig_dict['squares'] += 1
            else:
                shape_name = 'Rectangle'
                _fig_dict['rectangles'] += 1
        elif len(approx) > 10:
            shape_name = 'Circle'
            _fig_dict['circles'] += 1
        else:
            # print("Detected unrecognized shape!")
            # continue
            shape_name = 'Unknown'
            # raise Exception("Unexpected figure (unseen in the training set) with {len(approx)} sides found.")

        if _visualize:
            cv2.putText(image, f"{colors_found[i]} {shape_name.lower()}",
                        __retrieve_coords(approx), cv2.FONT_HERSHEY_DUPLEX, 1, BLACK, 1)

    if _visualize:
        # displaying the image with the detected shapes onto the screen
        # drawing the outer-edges onto the image
        cv2.drawContours(image, contours, contourIdx=-1, color=BLACK,
                         thickness=4, lineType=cv2.LINE_AA)
        cv2.imshow("shapes_detected", cv2.resize(image, (700, 700)))
        cv2.waitKey(0)

    return _fig_dict


def __retrieve_coords(approx) -> Tuple:
    # retrieving coordinates of the contour so that we can put text over the shape.
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + (w / 3))
    y_mid = int(y + (h / 1.5))

    # variables used to display text on the final image
    return x_mid, y_mid


def __is_square(approx, tolerance: int = None) -> bool:
    # retrieving coordinates of the contour
    x, y, w, h = cv2.boundingRect(approx)

    if tolerance is None:
        tolerance = min(w // 10, h // 10)

    return abs(w - h) < tolerance


if __name__ == "__main__":
    print("Custom",
          retrieve_figures('train/custom.png', _visualize=True))
    # print("Triangles", retrieve_figures('train/triangles.png', _visualize=True))
    # print("Quadrats", retrieve_figures('train/quadrats.png'))
    # print("Rectangles", retrieve_figures('train/rectangles.png'))
    # print("Cercles", retrieve_figures('train/cercles.png'))
