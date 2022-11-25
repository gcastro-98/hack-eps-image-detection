import sys
from baseline import retrieve_figures


def main() -> None:
    image_path: str = sys.argv[1]
    figures_dict: dict = retrieve_figures(image_path)
    print_result(figures_dict)


def print_result(fig_dict: dict) -> None:
    text: str = \
        f"""
        ----------- COLOR ------------
        
        Vermelles:      {fig_dict['red']}
        Verdes:         {fig_dict['green']}
        Blaves:         {fig_dict['blue']}
        
        ----------- FORMES -----------
        
        Triangles:      {fig_dict['triangles']}
        Quadrats:       {fig_dict['squares']}
        Rectangles:     {fig_dict['rectangles']}
        Cercles:        {fig_dict['circles']}
        
        ------ LOGOS LLEIDAHACK ------
        
        Logos:          {fig_dict['logos']}
        
        """
    print(text)


# def argument_parser():
#     """
#     Function to parse the arguments passed by CLI. The arguments are:
#     --image: The path to the input image we want to apply object detection to
#     --model: The type of PyTorch object detector we’ll be using (Faster R-CNN + ResNet, Faster R-CNN + MobileNet, or RetinaNet + ResNet)
#     --labels: The path to the COCO labels file, containing human-readable class labels
#     --confidence: Minimum predicted probability to filter out weak detections
#     :return:
#     """
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", type=str, required=True, help="path to the input image")
#     ap.add_argument("-m", "--model", type=str, default="frcnn-resnet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"], help="name of the object detection model")
#     ap.add_argument("-l", "--labels", type=str, default="coco_classes_index.pickle", help="path to file containing list of categories in COCO dataset")
#     ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
#     args = vars(ap.parse_args())
#     return args

# --model frcnn-resnet


if __name__ == '__main__':
    main()
