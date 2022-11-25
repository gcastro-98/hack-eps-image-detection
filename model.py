import cv2
import numpy as np
from typing import Tuple, Union
BLACK: tuple = (0, 0, 0)


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


def __define_color_bounds(_type: str, color: str) \
        -> Tuple[Union[float, list], Union[float, list]]:

    if _type == 'hsv':
        lower = {'red': ([0, 50, 20]), 'green': ([35, 50, 20]),
                 'blue': ([80, 50, 20]), 'orange': ([8, 50, 90])}
        upper = {'red': ([6, 255, 255]), 'green': ([79, 255, 255]),
                 'blue': ([149, 255, 255]), 'orange': ([34, 255, 255])}
        __assert_empty_intersection(lower, upper)
    else:
        raise KeyError(f"Unrecognized color code: {_type}; "
                       f"or not yet implemented.")

    lower, upper = lower[color], upper[color]

    if color == 'red':
        return [lower, ([150, 50, 20])], [upper, ([255, 255, 255])]

    return lower, upper


def __create_mask(image, color: str, _type: str = 'hsv'):
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    if _type == 'hsv':
        lower, upper = __define_color_bounds('hsv', color)
        _c_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    elif _type == 'rgb':
        lower, upper = __define_color_bounds('rgb', color)  # not implemented
        _c_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    else:
        raise KeyError(f'Color type not recognized: {_type}')

    kernel = np.ones((2, 2), np.uint8)

    if color != 'red':
        mask = cv2.inRange(_c_image, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1 if color != 'orange' else 20)
    else:
        mask = __red_mask(_c_image, lower, upper)
    mask = cv2.erode(mask, np.ones((10, 10), np.uint8))
    return mask


def __red_mask(_c_image, lower, upper):
    kernel = np.ones((2, 2), np.uint8)
    _mask_1 = cv2.inRange(_c_image, np.array(lower[0]), np.array(upper[0]))
    _mask_2 = cv2.inRange(_c_image, np.array(lower[1]), np.array(upper[1]))
    mask = cv2.bitwise_or(_mask_1, _mask_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def detect_shapes_and_colors(image_path: str, _visualize: bool = False,
                             _see_mask: bool = False) -> dict:
    image = __load_image(image_path)
    _fig_dict = {'triangles': 0, 'squares': 0, 'rectangles': 0, 'circles': 0,
                 'red': 0, 'green': 0, 'blue': 0, 'logos': 0}

    contours = []  # obtained after a mask
    shapes_list = []
    colors_found = []

    for key in ['red', 'green', 'blue', 'orange']:
        mask = __create_mask(image, key)

        if _see_mask:
            cv2.imshow(f"{key.capitalize()} mask",
                       cv2.resize(mask, (700, 700)))
            cv2.waitKey(0)

        # the second output is the hierarchy: not used
        _contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for _contour in _contours:
            true_shape = __find_shape_from_contour(_contour)
            colors_found.append(key)
            shapes_list.append(true_shape)
            contours.append(_contour)  # not used

    # Iterating through each contour to retrieve coordinates of each shape
    for i, approx in enumerate(shapes_list):
        if colors_found[i] == 'orange':
            shape_name = 'Logo'
            _fig_dict['logos'] += 1

        elif len(approx) == 3:
            shape_name = 'Triangle'
            _fig_dict['triangles'] += 1
            _fig_dict[colors_found[i]] += 1
        elif len(approx) == 4:
            if __is_square(approx):
                shape_name = 'Square'
                _fig_dict['squares'] += 1
            else:
                shape_name = 'Rectangle'
                _fig_dict['rectangles'] += 1
            _fig_dict[colors_found[i]] += 1
        elif len(approx) > 10:
            shape_name = 'Circle'
            _fig_dict['circles'] += 1
            _fig_dict[colors_found[i]] += 1
        else:
            # shape_name = 'Unknown'
            continue

        if _visualize:
            cv2.putText(image, f"{colors_found[i]} {shape_name.lower()}",
                        __retrieve_coords(approx), cv2.FONT_HERSHEY_DUPLEX,
                        1, BLACK, 1)

    if _visualize:
        # displaying the image with the detected shapes onto the screen
        # drawing the outer-edges onto the image
        cv2.drawContours(image, contours, contourIdx=-1, color=BLACK,
                         thickness=4, lineType=cv2.LINE_AA)
        cv2.imshow("shapes_detected", cv2.resize(image, (700, 700)))
        cv2.waitKey(0)

    return _fig_dict


def __find_shape_from_contour(contour):
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    return approx


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
