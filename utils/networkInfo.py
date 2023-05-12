"""
name : networkInfo.py
usage : write network information into txt file
author : Bo-Yuan You
Date : 2023-04-19

"""

from prettytable import PrettyTable

def countParameters(model):
    """
        Return table, number of total parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    totalParams = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        totalParams+=params
    return table, totalParams
