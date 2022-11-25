import cv2
BLACK: tuple = (0, 0, 0)


def retrieve_figures(image_path: str, _visualize: bool = False) -> dict:
    fig_dict: dict = {}
    fig_dict.update(_detect_shapes(image_path, _visualize))
    # TODO: change and actually detect colors and the logo
    fig_dict.update({'red': 0, 'green': 0, 'blue': 0, 'logos': 0})
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


def _detect_shapes(image_path: str, _visualize: bool = False) -> dict:
    image = __load_image(image_path)
    _fig_dict = {'triangles': 0, 'squares': 0, 'rectangles': 0, 'circles': 0}
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting to gray image

    # setting threshold value to get new image: depending on how
    # dark the pixel is, the threshold value will convert the pixel
    # to either black or white (0 or 1)).
    _, thresh_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)

    # retrieving outer-edge coordinates in the new threshold image
    contours, hierarchy = cv2.findContours(
        thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterating through each contour to retrieve coordinates of each shape
    for i, contour in enumerate(contours):
        if i % 2 == 0:
            # TODO: figure out why there 2 contours for each
            continue

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        if len(approx) == 3:
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
            print("Detected unrecognized shape!")
            continue
            # raise Exception("Unexpected figure (unseen in the training set) with {len(approx)} sides found.")

        if _visualize:
            # retrieving coordinates of the contour so that we can put text over the shape.
            x, y, w, h = cv2.boundingRect(approx)
            x_mid = int(x + (w / 3))
            y_mid = int(y + (h / 1.5))

            # variables used to display text on the final image
            coords = (x_mid, y_mid)
            colour = BLACK
            font = cv2.FONT_HERSHEY_DUPLEX

            # if the length is not any of the above,
            # we will guess the shape/contour to be a circle.
            cv2.putText(image, shape_name, coords, font, 1, colour, 1)

    if _visualize:
        # displaying the image with the detected shapes onto the screen
        # drawing the outer-edges onto the image
        cv2.drawContours(image, contours, contourIdx=-1, color=BLACK,
                         thickness=4, lineType=cv2.LINE_AA)
        cv2.imshow("shapes_detected", image)
        cv2.waitKey(0)

    return _fig_dict


def __is_square(approx, tolerance: int = None) -> bool:
    # retrieving coordinates of the contour
    x, y, w, h = cv2.boundingRect(approx)

    if tolerance is None:
        tolerance = min(w // 10, h // 10)

    return abs(w - h) < tolerance


if __name__ == "__main__":
    print("Triangles", retrieve_figures('train/triangles.png'))
    print("Quadrats", retrieve_figures('train/quadrats.png'))
    print("Rectangles", retrieve_figures('train/rectangles.png'))
    print("Cercles", retrieve_figures('train/cercles.png', _visualize=True))
