from argparse import ArgumentParser
from model import detect_shapes_and_colors


def _arg_parser():
    ap = ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="Path to the input image")
    ap.add_argument("-v", "--visualize", type=bool, default=False,
                    choices=[True, False],
                    help="If True, then the image is displayed labeled with "
                         "the found colors and shapes")
    ap.add_argument("-m", "--mask", type=bool, default=False,
                    choices=[True, False],
                    help="If True, then the masks for each color (R, G, B and"
                         "the logo, 'orange') are displayed")
    args = vars(ap.parse_args())
    return args


def main() -> None:
    # image_path: str = sys.argv[1]
    figures_dict: dict = detect_shapes_and_colors(
        *[_v for _v in _arg_parser().values()])
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


if __name__ == '__main__':
    main()
