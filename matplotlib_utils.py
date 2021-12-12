import matplotlib as mpl


def set_matplotlib_properties():
    # TODO not sure if this is the correct font?
    # use latex font
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['figure.figsize'] = (8, 5)
    mpl.rcParams['legend.handletextpad'] = 0.3
    mpl.rcParams['legend.handlelength'] = 1.0
    mpl.rcParams['legend.handleheight'] = 0.2
    mpl.rcParams['legend.labelspacing'] = 0.1
    mpl.rcParams['legend.borderaxespad'] = 0.2
    mpl.rcParams['legend.borderpad'] = 0.3