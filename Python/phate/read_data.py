import scipy.io as sio
import numpy

def import_single_cell_file(data_file, file_type):
    print("Importing data...")

    file_type = file_type.lower()

    if file_type == 'mtx':
        import scipy.io as sio
        M = sio.mmread(data_file)
    elif file_type == 'csv':
        import numpy as np
        M = np.loadtxt(data_file, delimiter=',')
    elif file_type == 'tsv':
        import numpy as np
        M = np.loadtxt(data_file, delimiter='\t')
    elif file_type == 'fcs':
        raise NotImplementedError("FCS files are not currently supported. Please post to the GitHub if you're interested in this function.")
    else:
        raise ValueError("Supported files types are ['mtx', 'csv', 'tsv', 'mat', 'fcs']")

    print("Imported data matrix with %s cells and %s genes..."%(M.shape[0], M.shape[1]))
    return(M)
