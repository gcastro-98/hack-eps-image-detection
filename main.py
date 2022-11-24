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



